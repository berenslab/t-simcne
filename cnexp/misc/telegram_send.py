import inspect
import json
import subprocess
import sys
from pathlib import Path

import telegram
from PIL.PngImagePlugin import PngImageFile


def get_token_chat_id():
    modfile = Path(inspect.getfile(TelegramBot))
    with open(modfile.parent / "telegram.json") as f:
        rc = json.load(f)

    return rc


class TelegramBot:
    def __init__(self, token=None, chat_id=None):
        modfile = Path(inspect.getfile(type(self)))
        with open(modfile.parent / "telegram.json") as f:
            self.rc = json.load(f)

        self.token = token if token is not None else self.rc["token"]
        self.chat_id = chat_id if chat_id is not None else self.rc["chat_id"]
        self.bot = telegram.Bot(token=self.token)

    def send(self, fname):
        fname = Path(fname)
        ext = fname.suffix[1:]
        if ext == "mp4":
            return self.send_mp4(fname)
        elif ext == "png":
            return self.send_png(fname)
        elif ext == "txt" or ext == "md":
            return self.send_txt(fname)
        else:
            raise RuntimeError(
                "File ext '{}' not understood, file {} not sent".format(
                    ext, fname
                )
            )

    def send_mp4(self, fname):
        # how to get video info example taken from
        # https://github.com/kkroening/ffmpeg-python/blob/master/examples/video_info.py
        import ffmpeg

        try:
            probe = ffmpeg.probe(fname)
        except ffmpeg.Error as e:
            print(e.stderr, file=sys.stderr)
            sys.exit(1)

        video_stream = next(
            (
                stream
                for stream in probe["streams"]
                if stream["codec_type"] == "video"
            ),
            None,
        )

        if video_stream is None:
            print("No video stream found", file=sys.stderr)
            sys.exit(1)

        width = int(video_stream["width"])
        height = int(video_stream["height"])
        duration = float(video_stream["duration"])
        # num_frames = int(video_stream["nb_frames"])

        ffmpeg_proc = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "warning",
                "-i",
                fname,
                "-c",
                "copy",
                "-map_metadata",
                "0",
                "-map_metadata:s:v",
                "0:s:v",
                "-f",
                "ffmetadata",
                "-",
            ],
            encoding="utf-8",
            stdout=subprocess.PIPE,
        )

        # filter out the comment field from the video metadata
        lines = iter([str(line) for line in ffmpeg_proc.stdout.split("\n")])
        for line in lines:
            print(line)
            if "comment=" in line:
                comment = line[len("comment=") :]
                # assumption: there will not be an escaped backslash at
                # the end of the comment string.  In this case it will
                # actually bleed into the next line, which isn't too bad,
                # just some (irrelevant) extra information.
                while line[-1] == "\\":
                    try:
                        line = next(lines)
                    except StopIteration:
                        break
                    finally:
                        comment = comment[:-1]
                    comment = comment[:-1] + "\n" + line

                break

        try:
            comment
        except NameError:
            comment = f"`{fname}`"  # make sure that the variable exists

        with open(fname, "rb") as f:
            return self.bot.send_animation(
                chat_id=self.chat_id,
                animation=f,
                width=width,
                height=height,
                duration=duration,
                parse_mode="Markdown",
                caption=comment,
            )

    def send_png(self, fname):
        png = PngImageFile(fname)

        png.load()  # load metadata
        with open(fname, "rb") as f:
            return self.bot.send_photo(
                chat_id=self.chat_id,
                photo=f,
                parse_mode="Markdown",
                caption=png.info.get("Comment", "`{}`".format(fname)),
            )

    def send_txt(self, fname):
        filename = f"`{fname}`\n`---`\n"
        filecontent = fname.read_text()

        text = filename + filecontent
        return self.bot.send_message(
            chat_id=self.chat_id, text=text, parse_mode="Markdown"
        )


if __name__ == "__main__":
    bot = TelegramBot()
    bot.send(sys.argv[1])
