#!/usr/bin/env python
import multiprocessing
from typing import List, Optional
import PIL.Image
import time
import os
import asyncio
from io import BytesIO
from pathvalidate import sanitize_filename

import interactions

import config
import scheduler

bot = interactions.Client(token=config.DISCORD_TOKEN)

@bot.command(
    name="text2img",
    description="Convert text to image",
    options = config.default_options,
)
async def text2img(
    ctx: interactions.CommandContext,
    text: str,
    count: int=1,
    width: int=config.DEFAULT_HEIGHT,
    height: int=config.DEFAULT_WIDTH,
    seed: int=config.DEFAULT_SEED,
    nsfw: bool=config.NSFW_GLOBAL_ENABLE,
):
    print("Command /text2img received")
    # Input validation
    if height % 8 != 0 or width % 8 != 0:
        await ctx.send(f"Height and width must be divisible by 8", ephemeral=True)
        return
    if height > config.MAX_HEIGHT:
        await ctx.send(f"Height must be ≤{config.MAX_HEIGHT}", ephemeral=True)
        return
    if width > config.MAX_WIDTH:
        await ctx.send(f"Width must be ≤{config.MAX_WIDTH}", ephemeral=True)
        return
    if count > config.MAX_COUNT:
        await ctx.send(f"Count must be ≤${config.MAX_COUNT}", ephemeral=True)
        return

    embed = default_embed(text, count, width, height, seed, nsfw)
    attachments = []

    await ctx.send(embeds=embed)
    try:

        # throttle is used to not spam edit the message.
        t = None
        async def throttle_edit():
            nonlocal t, ctx, embed
            now = time.time()
            if t is None or now - t > config.DEFAULT_THROTTLE_SECONDS:
                await ctx.edit(embeds=embed)
                t = now

        async for status in scheduler.schedule(text, width=width, height=height, seed=seed, sfw=not nsfw, samples=count):
            embed.fields[-1].value = str(status)
            if status.images is None:
                await throttle_edit()
            else:
                attachments=images2attachments(status.images, ctx.user, text, seed)

    except asyncio.TimeoutError:
        await ctx.edit(f"Could not generate image{'s' if count != 1 else ''}.", embeds=None)
        await ctx.send("Timed out waiting for Stable Diffusion, this usually happens during startup when the bot is seeding the model cache for the first time. Try again soon.", ephemeral=True)
        return
    except:
        await ctx.edit(f"Could not generate image{'s' if count != 1 else ''}.", embeds=None)
        await ctx.send("Something went wrong. Try again later.", ephemeral=True)
        raise

    if len(attachments) == 0:
        await ctx.edit(f"Could not generate image{'s' if count != 1 else ''}.", embeds=None)
        await ctx.send(f"NSFW image{'s' if count != 1 else ''} detected." + config.NSFW_WARNING, embeds=None, ephemeral=True)
    elif len(attachments) < count:
        await ctx.send(f"Some NSFW images have been excluded." +config.NSFW_WARNING, embeds=None, ephemeral=True)
    else:
        await ctx.message.edit(text, embeds=None, files=attachments)

# Constructs the default fields for the embed
def default_embed(text, count, width, height, seed, nsfw):
    fields=[interactions.EmbedField(name="Text", value=text)]

    if width != config.DEFAULT_WIDTH or height != config.DEFAULT_HEIGHT:
        fields.append(interactions.EmbedField(name="Width", value=width, inline=True))
        fields.append(interactions.EmbedField(name="Height", value=height, inline=True))
    if seed != config.DEFAULT_SEED:
        fields.append(interactions.EmbedField(name="Seed", value=seed, inline=True))
    if count != 1:
        fields.append(interactions.EmbedField(name="Count", value=count, inline=True))

    fields.append(interactions.EmbedField(name=f"Progress", value="Queuing...", inline=False))
    return interactions.Embed(fields=fields)

def images2attachments(images: List[PIL.Image.Image], user, prompt, seed):
    attachments = []
    for i, img in enumerate(images):
        path = f"output/{user.username}#{user.discriminator}/"
        os.makedirs(path, exist_ok=True)
        filename = f"{path}/{prompt}_steps({config.DEFAULT_STEPS})_scale({config.DEFAULT_SCALE})_seed({seed})_{i}.{config.FILE_FORMAT}"

        buf = BytesIO()
        img.save(buf, format=config.FILE_FORMAT)
        if config.SAVE_IMAGE_TO_DISK:
            img.save(filename)
        buf.seek(0)
        attachments.append(interactions.api.models.misc.File(filename, buf))
    return attachments

if __name__ == '__main__':
    # Run stable diffusion in dedicated process. This makes managing GPU memory
    # easier, as memory errors can be cleaned up by killing the sub process.
    multiprocessing.set_start_method('spawn')

    scheduler = scheduler.Scheduler(
        token=config.HUGGINGFACE_TOKEN,
        model=config.SD_MODEL,
        revision=config.SD_REVISION,
        attention_slicing=config.SD_ATTENTION_SLICING,
        default_seed=config.DEFAULT_SEED,
    )
    with scheduler:
        bot.start()
