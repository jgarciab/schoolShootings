#!/bin/bash
for var in "$@"
do
	s=${var##*/}
	base=${s%.svg}
	echo 'converting' $var
	inkscape -f $var -A $base.pdf
	echo 'compressing' $var
	ghostscript -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$base-optimized.pdf $base.pdf

	echo 'done!'
done

