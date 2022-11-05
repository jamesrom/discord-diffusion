import os
import interactions

# Default settings
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
DEFAULT_HEIGHT = os.getenv('DEFAULT_HEIGHT', 512)
DEFAULT_SEED = os.getenv('DEFAULT_SEED', None)
DEFAULT_WIDTH = os.getenv('DEFAULT_WIDTH', 512)
FILE_FORMAT = os.getenv("FILE_FORMAT", "jpeg").lower()
MAX_COUNT = int(os.getenv('MAX_IMAGE_COUNT', 5))
MAX_HEIGHT = os.getenv('MAX_HEIGHT', 2048)
MAX_WIDTH = os.getenv('MAX_WIDTH', 2048)
NSFW_GLOBAL_ENABLE = os.getenv("NSFW_GLOBAL_ENABLE", 'False').lower() in ('true', '1', 't')
NSFW_IN_AGE_RESTRICTED_CHANNELS = os.getenv("NSFW_IN_AGE_RESTRICTED_CHANNELS", 'True').lower() in ('true', '1', 't')
NSFW_USER_OVERRIDE = os.getenv("NSFW_USER_OVERRIDE", 'False').lower() in ('true', '1', 't')
SAVE_IMAGE_TO_DISK = os.getenv("SAVE_IMAGE_TO_DISK", 'True').lower() in ('true', '1', 't')
SD_ATTENTION_SLICING = os.getenv("SD_ATTENTION_SLICING", 'False').lower() in ('true', '1', 't')
SD_MODEL = os.getenv("SD_MODEL", "CompVis/stable-diffusion-v1-4")
SD_REVISION = os.getenv("SD_REVISION", "main")

# Validation
assert len(HUGGINGFACE_TOKEN) > 0, "env HUGGINGFACE_TOKEN not set"
assert len(DISCORD_TOKEN) > 0, "env DISCORD_TOKEN not set"
assert FILE_FORMAT in ('jpeg', 'png'), "env FILE_FORMAT must be 'jpeg' or 'png'"
assert SD_REVISION in ('main', 'fp16'), "env REVISION must be 'main' or 'fp16'"

# Constants
DEFAULT_STEPS = 50
DEFAULT_SCALE = 7.5
DEFAULT_THROTTLE_SECONDS = 3 # The size of the time window to discard status
                             # updates, used to prevent spamming too much.

NSFW_WARNING = """Here's some things you can try:
 • Use a different seed
 • Use this command in an [age-restricted channel](https://support.discord.com/hc/en-us/articles/115000084051-Age-Restricted-Channels-and-Content).
""" + " • The server admin has allowed you to override this warning. Set `nsfw=True` and try again." if NSFW_USER_OVERRIDE else ""


# Default /text2img command options
default_options = [
    interactions.Option(
        name="text",
        description="Describe the image.",
        type=interactions.OptionType.STRING,
        required=True,
    ),
    interactions.Option(
        name="width",
        description=f"Image width. Must be divisible by 8. (default {DEFAULT_WIDTH})",
        type=interactions.OptionType.INTEGER,
        required=False,
    ),
    interactions.Option(
        name="height",
        description=f"Image height. Must be divisible by 8. (default {DEFAULT_HEIGHT})",
        type=interactions.OptionType.INTEGER,
        required=False,
    ),
    interactions.Option(
        name="seed",
        description=f"Seed for random numbers. Use this to get different results with the same prompt.",
        type=interactions.OptionType.INTEGER,
        required=False,
    ),
]

if MAX_COUNT > 1:
    default_options.append(
        interactions.Option(
            name="count",
            description=f"How many images to generate (1-{MAX_COUNT})",
            type=interactions.OptionType.INTEGER,
            required=False,
        )
    )

if NSFW_USER_OVERRIDE and not NSFW_GLOBAL_ENABLE:
    default_options.append(
        interactions.Option(
            name="nsfw",
            description=f"Allow NSFW images to be generated (this is enabled by default in Age-restricted channels)",
            type=interactions.OptionType.BOOLEAN,
            required=False,
        )
    )
