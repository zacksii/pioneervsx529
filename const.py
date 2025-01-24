# Domain for the integration
DOMAIN = "pioneervsx529"

# Configuration options
CONF_SOURCES = "sources"
CONF_HOST = "host"
CONF_PORT = "port"

# Maximum volume (to scale the volume to a 0..1 range)
MAX_DB = 161       # Maximum volume in dB
MIN_DB = 0     # Minimum volume in dB
MAX_VOLUME = 80  # Maximum device volume scale



# Define the default port in case the user doesn't specify one
DEFAULT_PORT = 8102


# Default sources schema
DEFAULT_SOURCES = {
    "CD": "01",
    "FM tuner": "02",
    "DVD": "04",
    "TV": "05",
    "SAT/CBL": "06",
    "No Device": "17",
    "HDMI": "19",
    "BD": "25",
    "ADP": "33",
    "Net Radio": "38",
    "M Server": "44",
    "Favorite": "45",
    "MHL": "48",
    "Game": "49",
}