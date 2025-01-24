import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.const import CONF_HOST, CONF_PORT
import logging
import asyncio
from .const import DOMAIN, DEFAULT_SOURCES, MAX_VOLUME, DEFAULT_PORT

_LOGGER = logging.getLogger(__name__)

# Generate dynamic schema for sources
def generate_sources_schema(defaults):
    return {
        vol.Optional(key, default=default): str
        for key, default in defaults.items()
    }

SOURCES_SCHEMA = generate_sources_schema(DEFAULT_SOURCES)

class PioneerVSX529ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Pioneer VSX-529 integration."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_POLL

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}

        _LOGGER.debug("Starting config flow for Pioneer VSX-529")

        if user_input is not None:
            host = user_input[CONF_HOST]
            port = user_input[CONF_PORT]

            # Sanitize sources to ensure valid strings
            sanitized_sources = {}
            for key, default_value in DEFAULT_SOURCES.items():
                value = user_input.get(key, default_value).strip()
                if not value:
                    errors["base"] = "invalid_sources"
                    break
                sanitized_sources[key] = value

            if errors:
                _LOGGER.error("Invalid sources configuration.")
                return self.async_show_form(
                    step_id="user", data_schema=vol.Schema(SOURCES_SCHEMA), errors=errors
                )

            _LOGGER.debug("Received user input: host=%s, port=%d, sources=%s", host, port, sanitized_sources)

            # Validate the IP and Port
            if not await self._is_valid_device(host, port):
                _LOGGER.error("Connection test failed for %s:%d", host, port)
                errors["base"] = "cannot_connect"
            else:
                _LOGGER.debug("Connection test succeeded for %s:%d", host, port)
                return self.async_create_entry(
                    title=f"Pioneer VSX-529 ({host})",
                    data={
                        CONF_HOST: host,
                        CONF_PORT: port,
                        "sources": sanitized_sources,
                        "volume_step": user_input.get("volume_step", 1),
                    },
                )

        # Default form schema
        data_schema = vol.Schema(
            {
                vol.Required(CONF_HOST): str,
                vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
                vol.Optional("volume_step", default=1): vol.All(vol.Coerce(int), vol.Range(min=1, max=MAX_VOLUME)),
                **SOURCES_SCHEMA,
            }
        )

        _LOGGER.debug("Displaying config flow form with errors: %s", errors)
        return self.async_show_form(step_id="user", data_schema=data_schema, errors=errors)

    async def _is_valid_device(self, host, port):
        """Check if the device is reachable and valid."""
        reader = None
        writer = None
        try:
            _LOGGER.debug("Attempting to connect to Pioneer VSX-529 at %s:%d", host, port)
            reader, writer = await asyncio.open_connection(host, port)
            _LOGGER.debug("Connection established, sending test command")
            writer.write(b"?P\r")  # Power state query as a basic test
            await writer.drain()
            _LOGGER.debug("Test command sent, waiting for response")
    
            while True:
                response = await asyncio.wait_for(reader.readuntil(b'\r'), timeout=10)
                response = response.decode().strip()
                _LOGGER.debug("Received response: %s", response)
    
                if response == "R":
                    _LOGGER.debug("Keep-alive response received, ignoring.")
                    continue  # Ignore keep-alive messages and keep waiting
    
                if "PWR" in response:
                    _LOGGER.debug("Valid response received from device: %s", response)
                    return True  # Device is valid
    
                _LOGGER.error("Unexpected response from device: %s", response)
                break  # Exit if an invalid response is received
    
        except asyncio.TimeoutError:
            _LOGGER.error("Connection test timed out for %s:%d", host, port)
        except Exception as e:
            _LOGGER.error("Failed to connect to Pioneer VSX-529 at %s:%d: %s", host, port, e)
        finally:
            # Ensure the connection is closed properly
            if writer:
                writer.close()
                await writer.wait_closed()
    
        return False  # Device is not valid

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow."""
        return PioneerVSX529OptionsFlowHandler(config_entry)

class PioneerVSX529OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options for Pioneer VSX-529."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self.config_entry = config_entry  # Store the config entry for use later

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            _LOGGER.debug("Updating options with user input: %s", user_input)
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={
                    **self.config_entry.data,
                    "volume_step": user_input.get("volume_step", self.config_entry.data.get("volume_step", 1)),
                    "sources": {key: user_input[key] for key in DEFAULT_SOURCES},
                },
            )
            return self.async_create_entry(title="", data={})

        _LOGGER.debug("Displaying options form for Pioneer VSX-529")
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        "volume_step", 
                        default=self.config_entry.data.get("volume_step", 1),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=MAX_VOLUME)),
                    **{
                        vol.Optional(
                            key, 
                            default=self.config_entry.data["sources"].get(key, default),
                        ): str
                        for key, default in DEFAULT_SOURCES.items()
                    },
                }
            ),
        )
