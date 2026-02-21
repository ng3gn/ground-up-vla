# NL2Bash Data

This repository clone only contains the raw data in all.cm and all.nl.

See the original at
https://github.com/TellinaTool/nl2bash/blob/master/data/

## Content

NL2Bash contains 12k one-line Linux shell commands used in practice and their
natural language descriptions provided by experts. The dataset includes 100+
commonly used shell utilities, providing a rich training and test bed for
automatically translating natural language into command lines.

The parallel data are stored with a source file `bash/\*.nl` and a target file
`bash/\*.cm`.

The entire corpus `bash/all.nl, bash/all.cm` are randomly splited into train,
dev, and test set with a ratio of `10:1:1`.

## Citation

If you use the data or source code in your work, please cite
```
@inproceedings{LinWZE2018:NL2Bash, 
  author = {Xi Victoria Lin and Chenglong Wang and Luke Zettlemoyer and Michael D. Ernst}, 
  title = {NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System}, 
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources
               and Evaluation {LREC} 2018, Miyazaki (Japan), 7-12 May, 2018.},
  year = {2018} 
}
```
