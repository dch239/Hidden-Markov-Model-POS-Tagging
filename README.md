This project is a hands-on experience on building HMM from scratch on part-of- speech tagging. Two decoding algorithms, Greedy Decoding and Viterbi Decoding, have been used to acheive 92% and 94% accuracies respectively.

The  Wall Street Journal section of the Penn Treebank dataset is used. In the folder named data, there are three files: train, dev and test. In the files of train and dev, we have the sentences with human-annotated part-of-speech tags. In the file of test, we have only the raw sentences that you need to predict the part-of-speech tags. The data format is that, each line contains three items separated by the tab symbol ‘\t’. The first item is the index of the word in the sentence. The second item is the word type and the third item is the corresponding part-of-speech tag. There is be a blank line at the end of each sentence.

The file vocab.txt, contains the vocabulary created on the training data. The format of the vocabulary file is that each line con- tains a word type, its index and its occurrences, separated by the tab symbol ‘\t’.

The file hmm.json, contains the emission and transition probabilities.

Two prediction files named greedy.out and viterbi.out, contains the predictions of my model on the test data with the greedy and viterbi decoding algorithms.

The PDF files contain the problem description and problem solutions with descriptions.
