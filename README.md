# Discord Diffusion

A self-hosted discord bot running [Stable Diffusion 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4).

## Setup

1. Get a discord bot token, connect it to your server. TODO: document
2. Get a huggingface token. TODO: document
3. Run the docker image:

```bash
docker run --rm -it --gpus=all \
    --volume $PWD\huggingface:/home/huggingface/.cache/huggingface \
    --volume $PWD\output:/home/huggingface/output \
    --env DISCORD_TOKEN=$BOT_TOKEN_HERE> \
    --env HUGGINGFACE_TOKEN=$HF_TOKEN_HERE \
    jamesrom/discord-diffusion:latest
```

## Configuration

Configure by environment variables:

| Variable        | Default | Description                               |
| --------------- | ------- | ----------------------------------------- |
| `DISCORD_TOKEN` | — | Discord token |
| `HUGGINGFACE_TOKEN` | — | Huggingface token |
| `DEFAULT_HEIGHT` | `512` | Must be divisible by 8 |
| `DEFAULT_WIDTH` | `512` | Must be divisible by 8 |
| `DEFAULT_SEED` | Random | Set the default seed for reproducability |
| `MAX_IMAGE_COUNT` | `5` | TODO |
| `GLOBAL_NSFW_ENABLE` | `false` | Bypass the built-in safety check |
| `FILE_FORMAT` | `jpeg` | Must be `jpeg` or `png` |
| `SAVE_IMAGE_TO_DISK` | `false` | Whether to save a copy of the generated image to disk |
| `REVISION` | `main` | The model revision to use |
| `ATTENTION_SLICING` | `false` | Use less memory at the cost of speed |

## Usage

Type `/text2img` and enter your prompt

![nolifer](docs/screenshot.png)

## Memory requirements

8GB recommended. Here's some things you can do if you see memory errors:

1. Set `REVISION=fp16`, this uses half-sized tensors (16 bit instead of 32 bit) and should reduce memory footprint by about half.
2. Enable attention slicing with `ATTENTION_SLICING=true`. This lowers memory requirements at the cost of speed.
3. Lower `DEFAULT_WIDTH` and `DEFAULT_HEIGHT`

## Building from source

TODO document

## More todos

- [ ] Move GPU work to isolated process for cleaner memory management
- [ ] Finer grained safety check configuration

## Thanks

Based on [Stable Diffusion in Docker](https://github.com/fboulnois/stable-diffusion-docker).
