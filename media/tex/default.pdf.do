# -*- mode: sh -*-
redo-ifchange $2.tex

OUTDIR=.out/$2/
rm -rf $OUTDIR
mkdir -p $OUTDIR
pdflatex -output-directory=$OUTDIR $2.tex >$OUTDIR/$2.out
ln $OUTDIR/$2.pdf $3
