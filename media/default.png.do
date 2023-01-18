# -*- mode: sh -*-

FILE=$2
redo-ifchange "$FILE.pdf"

if [ $(command -v pdftoppm) ]; then
    # leaving out a root name (second arg) when using `pdftoppm`
    # apparently writes the output to stdout.  This is not documented
    # anywhere in the man page as far as I can see.
    pdftoppm -r 300 -png -singlefile "$FILE.pdf" > $3
elif [ $(command -v convert) ]; then
    convert -density 1000 "$FILE.pdf" -resize 25% png:- > $3 2>/dev/null
else
    echo "No suitable command found for conversion.  Neither `pdftoppm' nor `convert' available." >&2
    exit 1
fi

if [ $(command -v exiftool) ]; then
   exiftool -overwrite_original -tagsfromfile "$FILE.pdf" $3 2>/dev/null >&2
fi
