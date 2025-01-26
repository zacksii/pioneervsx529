import time
import asyncio
import logging
import aiohttp
import re
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.components.media_player import MediaPlayerEntity
from homeassistant.components.media_player.const import MediaPlayerEntityFeature
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from .const import DOMAIN, MAX_VOLUME, MAX_DB, MIN_DB, DEFAULT_SOURCES

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the Pioneer media player from a config entry."""
    host = config_entry.data["host"]
    port = config_entry.data["port"]
    sources = config_entry.data.get("sources", {})

    _LOGGER.debug("Fetching system information for %s", host)
    try:
        system_info = await asyncio.wait_for(fetch_system_information(host), timeout=10)
        if not system_info:
            _LOGGER.error("Failed to fetch system information for %s. Entity will not be created.", host)
            return
    except asyncio.TimeoutError:
        _LOGGER.error("Timeout fetching system information for %s.", host)
        return

    device = PioneerDevice(hass, host, port, sources, system_info)

    async_add_entities([device], update_before_add=True)
    _LOGGER.debug("Entity added to Home Assistant: %s", device.entity_id)



async def _establish_connection(self):
    retries = 5
    delay = 1  # start with 1 second delay
    while retries > 0:
        try:
            self.telnet_connection = await asyncio.wait_for(self._connect(), timeout=10)
            return True
        except (asyncio.TimeoutError, ConnectionError):
            retries -= 1
            if retries > 0:
                _LOGGER.warning("Connection lost. Retrying in %d seconds.", delay)
                await asyncio.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                _LOGGER.error("Failed to reconnect after multiple attempts.")
                return False

async def send_command(self, command):
    if self.telnet_connection:
        try:
            await self.telnet_connection.write(command)
        except AttributeError:
            _LOGGER.error("Telnet connection lost when sending command.")
            await self._establish_connection()  # reconnect and retry
            return await self.send_command(command)
    else:
        _LOGGER.error("No active connection to send command.")



async def fetch_system_information(host):
    """Fetch system information from the device."""
    url = f"http://{host}/1000/system_information.asp"
    try:
        _LOGGER.debug("Fetching system information from %s", url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    _LOGGER.error("Failed to fetch system information from %s. HTTP status: %s", url, response.status)
                    return None
                html = await response.text()
#                _LOGGER.debug("System information HTML: %s", html)
                return parse_system_information(html)
    except Exception as e:
        _LOGGER.error("Error fetching system information: %s", e)
        return None

def parse_system_information(html):
    """Parse system information from device HTML."""
    try:
        matches = re.findall(r"write_field_2\((.*?)\);", html, re.DOTALL)
        if len(matches) < 2:
            _LOGGER.error("Second write_field_2 function not found in HTML.")
            return None

        args = matches[1]
        args = re.sub(r'"\s*\+\s*"', '', args)  # Remove JS string concatenation
        args = [arg.strip().strip('"').strip("'") for arg in args.split(",")]

#        _LOGGER.debug("Parsed system information arguments: %s", args)

        return {
            "software_version": args[0] if args[0] != "not available" else "Unknown",
            "friendly_name": args[1],
            "mac_address": args[4],
            "ip_address": args[6] if len(args) > 6 else "Unknown",
            "subnet_mask": args[7] if len(args) > 7 else "Unknown",
            "default_gateway": args[8] if len(args) > 8 else "Unknown",
            "primary_dns": args[9] if len(args) > 9 else "Unknown",
            "secondary_dns": args[10] if len(args) > 10 else "Unknown",
            "proxy_enabled": args[11].lower() == "on" if len(args) > 11 else False,
        }
    except Exception as e:
        _LOGGER.error("Error parsing system information: %s", e)
        return None

class PioneerDevice(MediaPlayerEntity):
    """Representation of a Pioneer VSX-529 media player."""

    _attr_supported_features = (
        MediaPlayerEntityFeature.SELECT_SOURCE
        | MediaPlayerEntityFeature.TURN_OFF
        | MediaPlayerEntityFeature.TURN_ON
        | MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.VOLUME_SET
        | MediaPlayerEntityFeature.VOLUME_STEP
    )

    def __init__(self, hass, host, port, sources, system_info):
        """Initialize the Pioneer device."""
        self.hass = hass
        self._host = host
        self._port = port
        self._sources = sources or {}
        self._source_name_to_number = self._sources
        self._source_number_to_name = {v: k for k, v in self._sources.items()}
        self._power = False
        self._volume = None
        self._muted = False
        self._selected_source = None
        self._system_info = system_info
        self._reader = None
        self._writer = None
        self._stop_listen = False
        self._state = "unknown"
        self.entity_id = f"media_player.{self.unique_id}"
        self._reader_lock = asyncio.Lock()
        self._command_lock = asyncio.Lock()
        self._update_lock = asyncio.Lock()

  
    

    async def send_command_with_retry(self, command: str, retries: int = 3):
        attempt = 0
        while attempt < retries:
            response = await self.send_command(command)
            if response:
                return response
            attempt += 1
            await asyncio.sleep(1)  # Delay between retries
        _LOGGER.error(f"Failed to send command after {retries} retries")
        return None
    
    
    async def _ensure_connection(self):
        """Ensure the connection to the device is active."""
        if self._reader and self._writer:
            return True  # Connection is already active
    
        _LOGGER.info("Reconnecting to Pioneer VSX-529 at %s:%s", self._host, self._port)
        try:
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
            _LOGGER.info("Connection re-established successfully.")
            return True
        except Exception as e:
            _LOGGER.error("Failed to reconnect to %s:%s: %s", self._host, self._port, e)
            self._reader, self._writer = None, None
            return False


    async def _close_connection(self):
        """Close the Telnet connection."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception as e:
                _LOGGER.warning("Error while closing connection: %s", e)
        self._reader, self._writer = None, None

    
    async def start_data_loop(self):
        """Start the persistent data reading loop after initialization."""
        _LOGGER.debug("Starting data reading loop.")
        asyncio.create_task(self._read_data())

    async def async_added_to_hass(self):
        """Handle entity being added to Home Assistant."""
        _LOGGER.debug("Entity added to Home Assistant: %s", self.entity_id)
    
        try:
            # Validate connection and fetch initial state
            await asyncio.gather(
                self.async_validate_connection(),
                self.fetch_initial_state(),
            )
        except Exception as e:
            _LOGGER.error("Error during async_added_to_hass: %s", e)
            self._state = "unavailable"
    
        self.async_write_ha_state()
    
        # Defer starting the data loop
        self.hass.loop.call_later(2, lambda: asyncio.create_task(self._read_data()))

    @property
    def state(self):
        """Return the state of the device."""
        if self._state == "unavailable":
            return STATE_UNAVAILABLE
        return STATE_ON if self._power else STATE_OFF
            
    async def async_will_remove_from_hass(self):
        """Handle entity removal."""
        _LOGGER.debug("Entity removed from Home Assistant: %s", self.entity_id)

    async def _state_check_loop(self):
        """Periodically check the state of the device."""
        while not self._stop_listen:
            if self._state == "unavailable":
                _LOGGER.info("Device unavailable, attempting reconnection...")
                if await self.async_validate_connection():
                    self._state = "off"  # Assume device starts off after reconnection
            await asyncio.sleep(30)  # Retry every 30 seconds

    async def async_update(self):
        """Fetch the latest state of the device."""
        _LOGGER.debug("Starting async_update for device %s", self.entity_id)
    
        try:
            async with self._update_lock:  # Prevent overlapping updates
                # Power State
                power_response = await self.telnet_command("?P")
                if power_response.startswith("PWR"):
                    self._power = power_response == "PWR2"
    
                # Volume Level
                volume_response = await self.telnet_command("?V")
                if volume_response.startswith("VOL"):
                    self._volume = self.scale_volume(volume_response)
    
                # Source
                source_response = await self.telnet_command("?F")
                if source_response.startswith("FN"):
                    self._selected_source = self.parse_source(source_response)
    
                self.async_write_ha_state()
        except asyncio.TimeoutError:
            _LOGGER.error("async_update timed out for %s.", self.entity_id)
        except Exception as e:
            _LOGGER.error("Unexpected error during async_update: %s", e)
      
   

    async def async_validate_connection(self):
        """Validate the Telnet connection and fetch the initial state."""
        start_time = time.time()
        _LOGGER.debug("Starting connection validation...")
        try:
            _LOGGER.debug("Validating Telnet connection to %s:%s", self._host, self._port)
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
            await self.telnet_command("?P")  # Power state
            await self.telnet_command("?V")  # Volume state
    
            await asyncio.sleep(0.5)  # Allow time for responses
            if self._reader.at_eof():
                raise ConnectionError("No response from device during validation.")
    
            # Process any initial responses
            while not self._reader.at_eof():
                data = await self._reader.readuntil(b'\r')
                self._parse_data(data.decode().strip())
    
            _LOGGER.info("Device connection validated successfully.")
            self.async_write_ha_state()
            return True
        except Exception as e:
            _LOGGER.error("Connection validation failed: %s", e)
            self._state = "unavailable"
            self.async_write_ha_state()
            return False
        finally:
            _LOGGER.debug("Connection validation completed in %.2f seconds", time.time() - start_time)

    async def fetch_initial_state(self):
        """Fetch initial state from the device."""
        await self.telnet_command("?P")
        await self.telnet_command("?V")
        await asyncio.sleep(0.5)  # Allow time for responses


  
    def _parse_data(self, data: str):
        """Parse data received from the device."""
        _LOGGER.debug("Raw data received from device: %s", data)
        try:
            if not data or len(data) < 3:  # Ignore very short or empty responses
                _LOGGER.warning("Received incomplete or empty data: %s", data)
                return

            if data.startswith("PWR"):
                self._power = data[3] == "0"
                self._state = STATE_ON if self._power else STATE_OFF
            elif data.startswith("VOL"):
                db_volume = int(data[3:6])
                self._volume = (db_volume - MIN_DB) / (MAX_DB - MIN_DB)
            elif data.startswith("MUT"):
                self._muted = data[3] == "0"
            elif data.startswith("FN"):
                source_number = data[2:4]
                self._selected_source = self._source_number_to_name.get(source_number, "Unknown")
            elif data.startswith("RGC"):
                _LOGGER.debug("RGC data received: %s", data)
                # Add specific parsing logic for RGC data here if needed.
            elif data == "R":
                _LOGGER.debug("Keep-alive response received.")
            else:
                _LOGGER.warning("Unknown or unexpected data: %s", data)
            self.async_write_ha_state()
        except Exception as e:
            _LOGGER.error("Error parsing data: %s", e)

                
    async def async_validate_connection(self):
        """Validate the Telnet connection and fetch the initial state."""
        try:
            _LOGGER.debug("Validating Telnet connection to %s:%s", self._host, self._port)
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
    
            # Send a test command and ensure the response is valid
            response = await self.telnet_command("?P")
            if "PWR" not in response:
                raise ValueError(f"Unexpected response during validation: {response}")
    
            _LOGGER.info("Device connection validated successfully.")
            self._state = STATE_OFF  # Assume device starts off
            self.async_write_ha_state()
            return True
        except Exception as e:
            _LOGGER.error("Connection validation failed: %s", e)
            self._state = "unavailable"
            self.async_write_ha_state()
            return False
    
    @property
    def device_info(self) -> DeviceInfo:
        """Return device information for Home Assistant."""
        _LOGGER.debug("Device info being passed: %s", self._system_info)
        return DeviceInfo(
            identifiers={(DOMAIN, self._host)},
            name=self._system_info.get("friendly_name", f"Pioneer VSX-529 ({self._host})"),
            manufacturer="Pioneer",
            model="VSX-529",
            sw_version=self._system_info.get("software_version", "Unknown"),
            connections=[("mac", self._system_info.get("mac_address", "Unknown"))],
        )

    @property
    def unique_id(self):
        """Return a unique ID for the entity."""
        return f"{self._host.replace('.', '_')}_{self._port}"

    @property
    def source_list(self):
        """Return a list of available input sources."""
        return list(self._source_name_to_number.keys())

    @property
    def available(self):
        """Return if the device is available."""
        return self._state != "unavailable"

    @property
    def device_class(self):
        """Return the class of this device."""
        return "receiver"

    async def set_volume_level(self, volume: float) -> None:
        """Set volume level incrementally, scaled from 0.0–1.0."""
        if self._volume is None:
            _LOGGER.error("Volume not initialized!")
            return
    
        if not self._reader or not self._writer:
            _LOGGER.warning("No active Telnet connection. Reconnecting...")
            await self._ensure_connection()
            if not self._reader or not self._writer:
                _LOGGER.error("Failed to establish a connection for volume adjustment.")
                return
    
        target_device_volume = round(volume * MAX_VOLUME)
        target_db_volume = self.scale_device_to_db(target_device_volume)
        current_db_volume = round(self._volume * (MAX_DB - MIN_DB) + MIN_DB)
    
        _LOGGER.debug(
            "Adjusting volume: slider=%.2f, current_db_volume=%d, target_db_volume=%d",
            volume, current_db_volume, target_db_volume,
        )
    
        attempts = 0
        max_attempts = 5
        delay = 0.2  # Start with a 200 ms delay between retries
    
        while current_db_volume != target_db_volume and attempts < max_attempts:
            try:
                if current_db_volume < target_db_volume:
                    await self.telnet_command("VU")
                    current_db_volume += 2  # Increment by 2 dB
                elif current_db_volume > target_db_volume:
                    await self.telnet_command("VD")
                    current_db_volume -= 2  # Decrement by 2 dB
    
                await asyncio.sleep(delay)
                delay *= 1.5  # Exponential backoff for retries
                attempts += 1
            except (ConnectionError, asyncio.TimeoutError) as e:
                _LOGGER.error(
                    "Error during volume adjustment on attempt %d: %s", attempts, e
                )
                await self._ensure_connection()  # Attempt to reconnect
            except Exception as e:
                _LOGGER.error("Unexpected error during volume adjustment: %s", e)
                break
    
        if current_db_volume != target_db_volume:
            _LOGGER.warning(
                "Volume adjustment incomplete. Current: %d dB, Target: %d dB",
                current_db_volume, target_db_volume,
            )
    
        # Update the volume state
        self._volume = (current_db_volume - MIN_DB) / (MAX_DB - MIN_DB)
        self.async_write_ha_state()
    
        if attempts >= max_attempts:
            _LOGGER.error("Failed to adjust volume after %d attempts.", max_attempts)
    
    
    def scale_device_to_db(self, device_volume: int) -> int:
        """Map device volume to dB."""
        if device_volume == 1:
            return 3
        return 3 + (device_volume - 1) * 2

    def scale_db_to_device(self, db_volume: int) -> int:
        """Map dB volume to device scale."""
        if db_volume <= 3:
            return 1
        return (db_volume - 3) // 2 + 1

    async def async_turn_on(self):
        """Turn the device on asynchronously."""
        try:
            await self.telnet_command("PO")
            self._state = STATE_ON
            self.async_write_ha_state()
        except RuntimeError as e:
            _LOGGER.error("Failed to turn on the device: %s", e)
    
    async def async_turn_off(self):
        """Turn the device off asynchronously."""
        try:
            await self.telnet_command("PF")
            self._state = STATE_OFF
            self.async_write_ha_state()
        except RuntimeError as e:
            _LOGGER.error("Failed to turn off the device: %s", e)
    

    async def mute_volume(self, mute: bool):
        """Mute or unmute the device."""
        try:
            await self.telnet_command("MO" if mute else "MF")  # Correctly await
            self._muted = mute
            self.async_write_ha_state()
        except Exception as e:
            _LOGGER.error("Error during mute/unmute command: %s", e)
    

    def select_source(self, source):
        """Select the input source."""
        source_number = self._source_name_to_number.get(source)
        if source_number:
            self.telnet_command(f"{source_number}FN")
            self._selected_source = source
            if self.hass:
                self.hass.loop.call_soon_threadsafe(self.async_write_ha_state)
                
    async def volume_up(self):
        """Volume up media player."""
        await self.telnet_command("VU")

    async def volume_down(self):
        """Volume down media player."""
        await self.telnet_command("VD")
        
    @property
    def source(self):
        """Return the currently selected input source."""
        return self._selected_source
        
    @property
    def is_volume_muted(self):
        """Return whether the volume is muted."""
        return self._muted
        
    @property
    def volume_level(self):
        """Return the volume level (0..1) for the slider."""
        return self._volume if self._volume is not None else 0.0

    @property
    def name(self):
        """Return the name of the device."""
        return self._system_info.get("friendly_name", f"Pioneer VSX-529 ({self._host})")

    async def telnet_command(self, command: str, attempts: int = 3) -> str:
        """Send a Telnet command and return the response."""
        for attempt in range(1, attempts + 1):
            async with self._command_lock:  # Prevent concurrent writes
                try:
                    _LOGGER.debug("Sending command: %s (attempt %d)", command, attempt)
                    self._writer.write(f"{command}\r".encode())
                    await self._writer.drain()
    
                    async with self._reader_lock:  # Prevent concurrent reads
                        response = await asyncio.wait_for(self._reader.readuntil(b"\r"), timeout=5)
                        response_str = response.decode().strip()
                        _LOGGER.debug("Received response for command '%s': %s", command, response_str)
                        return response_str
                except asyncio.TimeoutError:
                    _LOGGER.warning("Command '%s' timed out on attempt %d.", command, attempt)
                except Exception as e:
                    _LOGGER.error("Error sending command '%s' on attempt %d: %s", command, attempt, e)
                    await self._ensure_connection()  # Attempt to reconnect on error
    
        _LOGGER.error("Failed to execute command '%s' after %d attempts.", command, attempts)
        raise ConnectionError(f"Failed to execute command '{command}' after {attempts} attempts.")
    
    
    def parse_volume(self, volume_response: str) -> float:
        """Parse the volume response and return the normalized value (0.0 - 1.0)."""
        try:
            if volume_response.startswith("VOL"):
                raw_volume = int(volume_response[3:])  # Extract the numeric part
                return raw_volume / MAX_VOLUME  # Normalize based on MAX_VOLUME
        except ValueError:
            _LOGGER.error("Failed to parse volume response: %s", volume_response)
        return 0.0  # Default to mute level if parsing fails
        
    async def _read_data(self):
        """Continuously read data from the device."""
        _LOGGER.debug("Starting _read_data loop.")
        while not self._stop_listen:
            try:
                async with self._reader_lock:  # Ensure only one coroutine accesses the reader
                    if not self._reader or not self._writer:
                        _LOGGER.info("Reconnecting to Pioneer VSX-529 at %s:%s", self._host, self._port)
                        await self._ensure_connection()
                        continue
    
                    # Wait for data with a timeout
                    data = await asyncio.wait_for(self._reader.readuntil(b'\r'), timeout=10)
                    if data == b"R\r":  # Handle keep-alive response
                        _LOGGER.debug("Keep-alive response received.")
                        continue
                    self._parse_data(data.decode().strip())
            except asyncio.TimeoutError:
                _LOGGER.warning("Timeout waiting for data. Closing connection and retrying...")
                await self._close_connection()
                await asyncio.sleep(2)
            except Exception as e:
                _LOGGER.error("Connection error in _read_data: %s", e)
                await self._close_connection()
                await asyncio.sleep(2)
