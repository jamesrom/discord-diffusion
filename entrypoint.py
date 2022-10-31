#!/usr/bin/env python
import argparse, datetime, random, time
import torch
from torch import autocast
from diffusion import StableDiffusion
import workerqueue
import pathlib
import interactions
import os
import gc
from io import BytesIO
import asyncio
from PIL.Image import Image

# Default settings
DEFAULT_HEIGHT = os.getenv('DEFAULT_HEIGHT', 512)
DEFAULT_WIDTH = os.getenv('DEFAULT_WIDTH', 512)
DEFAULT_SEED = os.getenv('DEFAULT_SEED', None)
MAX_IMAGE_COUNT = int(os.getenv('MAX_IMAGE_COUNT', 5))
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
ALLOW_NSFW_USER_OVERRIDE = os.getenv("ALLOW_NSFW_USER_OVERRIDE", 'False').lower() in ('true', '1', 't')
GLOBAL_NSFW_ENABLE = os.getenv("GLOBAL_NSFW_ENABLE", 'False').lower() in ('true', '1', 't')
FILE_FORMAT = os.getenv("FILE_FORMAT", "jpeg").lower()
SAVE_IMAGE_TO_DISK = os.getenv("SAVE_IMAGE_TO_DISK", 'True').lower() in ('true', '1', 't')
REVISION = os.getenv("REVISION", "main")
ATTENTION_SLICING = os.getenv("ATTENTION_SLICING", 'False').lower() in ('true', '1', 't')

# Strings
NSFW_WARNING = """Here's some things you can try:
 • Use a different seed
 • Send this command in an Age-restricted channel.
"""

# Synchronization
mutex = asyncio.Lock()

assert len(HUGGINGFACE_TOKEN) > 0, "env HUGGINGFACE_TOKEN not set"
assert len(DISCORD_TOKEN) > 0, "env DISCORD_TOKEN not set"
assert FILE_FORMAT in ('jpeg', 'png'), "env FILE_FORMAT must be 'jpeg' or 'png'"
assert REVISION in ('main', 'fp16'), "env REVISION must be 'main' or 'fp16'"

diffuser = StableDiffusion(
    token=HUGGINGFACE_TOKEN,
    revision=REVISION,
    attention_slicing=ATTENTION_SLICING,
    default_seed=DEFAULT_SEED,
)
queue = workerqueue.DiffusionWorkerQueue(diffuser)
bot = interactions.Client(token=DISCORD_TOKEN)

# Default /text2img command options
default_options = [
    interactions.Option(
        name="text",
        description="Describe the image",
        type=interactions.OptionType.STRING,
        required=True,
    ),
    interactions.Option(
        name="width",
        description=f"Image width. Must be divisible by 8 (default {DEFAULT_WIDTH}",
        type=interactions.OptionType.INTEGER,
        required=False,
    ),
    interactions.Option(
        name="height",
        description=f"Image height. Must be divisible by 8 (default {DEFAULT_HEIGHT})",
        type=interactions.OptionType.INTEGER,
        required=False,
    ),
]

if MAX_IMAGE_COUNT > 1:
    default_options.append(
        interactions.Option(
            name="count",
            description=f"How many images to generate (1-{MAX_IMAGE_COUNT})",
            type=interactions.OptionType.INTEGER,
            required=False,
        )
    )

if ALLOW_NSFW_USER_OVERRIDE and not GLOBAL_NSFW_ENABLE:
    NSFW_WARNING += " • The server admin has allowed you to override this warning. Set `nsfw=True` and try again."
    default_options.append(
        interactions.Option(
            name="nsfw",
            description=f"Allow NSFW images to be generated (this is enabled by default in Age-restricted channels)",
            type=interactions.OptionType.BOOLEAN,
            required=False,
        )
    )

@bot.command(
    name="text2img",
    description="Convert text to image",
    options = default_options,
)
async def text2img(ctx: interactions.CommandContext, text: str, count: int=1, width: int=DEFAULT_HEIGHT, height: int=DEFAULT_WIDTH, nsfw: bool=GLOBAL_NSFW_ENABLE):
    if height % 8 != 0 or width % 8 != 0:
        await ctx.send(f"Height and width must be divisible by 8", ephemeral=True)
        return

    e = interactions.Embed(
        fields=[interactions.EmbedField(name=f"Dreaming...")],
    )

    # Callback for progress report
    async def cb(progress: workerqueue.QueueState):
        async with mutex:
            e.fields[0].value = progress.description()
            if ctx.message == None:
                await ctx.send(embeds=e)
            else:
                await ctx.edit(embeds=e)

    # Generate images
    try:
        files = await queue.run(text, progress_callback=cb, sfw=not nsfw, samples=count, width=width, height=height)
    except:
        await ctx.send("An error occurred. Try again later", embeds=None, ephemeral=True)
        raise
    msg = await ctx.edit('Done', embeds=None)

    if len(files) == 0:
        await ctx.send("NSFW image detected. "+NSFW_WARNING, embeds=None, ephemeral=True)
        return

    imgs = []
    for f in files:
        filename = f['name'] + "." + FILE_FORMAT
        img = f['image']
        img_buffer = BytesIO()
        img.save(img_buffer, format=FILE_FORMAT)
        if SAVE_IMAGE_TO_DISK:
            img.save(filename)
        img_buffer.seek(0)
        imgs.append(interactions.api.models.misc.File(filename, img_buffer))

    await msg.edit(text, files=imgs, embeds=None)

    if len(files) < count:
        await ctx.send("Some NSFW images have been excluded. "+NSFW_WARNING, embeds=None, ephemeral=True)

bot.start()
