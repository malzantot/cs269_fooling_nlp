#!/bin/bash
wget http://nlp.stanford.edu/data/glove.6B.zip
mkdir glove.6B
unzip glove.6B.zip -d glove.6B
rm -f glove.6B.zip
wget http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz
tar -xvf news20.tar.gz
rm -f news20.tar.gz
