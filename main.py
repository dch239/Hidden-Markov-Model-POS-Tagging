from collections import Counter
import json
from copy import deepcopy
import numpy as np
import string


# Task 1: Vocabulary Creation (20 points)
def create_vocabulary():
    # Read the file
    # Count the occurrence of words in a temporary dictionary 'd' first
    with open('data/train') as file:
        train_data = []
        temp = []
        word_counter = Counter()
        s = file.read().splitlines()
        for i in range(len(s)):
            if s[i] == '':
                train_data.append(temp)
                temp = []
                continue
            one_line = s[i].split('\t')
            word_counter[one_line[1]] += 1
            temp.append(one_line)
        train_data.append(temp)

    # Set a minimum threshold 'threshold'
    # If a word occurs less than 'threshold' times, remap it as '<unk>' in the final dictionary 'word_counter_final'
    threshold = 2
    word_counter_final = Counter()
    for k, v in word_counter.items():
        if v <= threshold:
            word_counter_final['<unk>'] += v
        else:
            word_counter_final[k] = v

    # Write the dictionary to vocab.txt
    with open('vocab.txt', 'w') as file:
        file.write(f"<unk>\t0\t{word_counter_final['<unk>']}\n")
        word_counter_final.pop('<unk>', None)
        i = 1
        for word, freq in word_counter_final.most_common():
            s = f"{word}\t{i}\t{freq}\n"
            i += 1
            file.write(s)
    return word_counter_final, train_data


# Task 2: Model Learning (20 points)
def model_learning(word_counter_final, data):
    tag_counter = Counter()
    emission_probabilities, transition_probabilities = Counter(), Counter()
    for i, sentence in enumerate(data):
        for j, word_desc in enumerate(sentence):
            if word_desc[1] not in word_counter_final:
                word_desc[1] = '<unk>'
            tag_counter[word_desc[2]] += 1
            emission_probabilities[(word_desc[2], word_desc[1])] += 1
            if j == 0:
                transition_probabilities[('<start>', word_desc[2])] += 1
            elif j == len(sentence):
                continue
            else:
                transition_probabilities[(sentence[j - 1][2], sentence[j][2])] += 1

    tag_counter['<start>'] = len(data)

    # Normalize the emission and transition probabilities
    for key, val in emission_probabilities.items():
        emission_probabilities[key] = val / tag_counter[key[0]]
    for key, val in transition_probabilities.items():
        transition_probabilities[key] = val / tag_counter[key[0]]

    # Write transition and emission probabilities to hmm.json
    js = {}
    t = {}
    e = {}
    for k, v in transition_probabilities.items():
        t[repr(k)] = v
    for k, v in emission_probabilities.items():
        e[repr(k)] = v
    js['transition'] = t
    js['emission'] = e
    with open("hmm.json", "w") as outfile:
        json.dump(js, outfile)

    return tag_counter, emission_probabilities, transition_probabilities


# Task 3: Greedy Decoding with HMM (30 points)
def greedy_decoding(data, word_counter_final, tag_counter, emission_probabilities, transition_probabilities):
    predicted_tags_greedy = []
    for sentence in deepcopy(data):
        word_tags = []
        prev_tag = '<start>'
        for word_desc in sentence:
            s = -1
            if word_desc[1] not in word_counter_final:
                word_desc[1] = '<unk>'
            for tag in tag_counter.keys():
                ep = emission_probabilities.get((tag, word_desc[1]), 0)
                tp = transition_probabilities.get((prev_tag, tag), 0)

                if (ep * tp) > s:
                    s = ep * tp
                    temp_tag = tag

            prev_tag = temp_tag
            word_tags.append(temp_tag)
        #         print(word_tags)
        predicted_tags_greedy.append(word_tags)
    return predicted_tags_greedy


# Task 4: Viterbi Decoding with HMM (30 Points)
def viterbi_decoding(data, word_counter_final, tag_counter, emission_count, transition_count):
    unique_tags = [x for x in tag_counter.keys() if x != "<start>"]

    def viterbi_one_sentence(sentence, unique_tags):
        n = len(sentence)
        sent = [elem[1] for elem in sentence]
        viterbi_matrix = np.zeros((len(unique_tags), len(sent)))
        backpointer_matrix = np.zeros((len(unique_tags), len(sent))).astype(int)
        for i, tag in enumerate(unique_tags):
            first_word = sent[0] if sent[0] in word_counter_final else '<unk>'
            viterbi_matrix[i][0] = transition_count.get(('<start>', tag), 0) * emission_count.get((tag, first_word),
                                                                                                  0)
        #         backpointer_matrix[i][0] = 0
        for i, word in enumerate(sent[1:]):
            for j, tag in enumerate(unique_tags):
                if word not in word_counter_final:
                    word = '<unk>'
                probs = [(viterbi_matrix[x][i] * transition_count.get((unique_tags[x], tag),
                                                                      0) * emission_count.get((tag, word), 0)) for x
                         in range(len(unique_tags))]
                viterbi_matrix[j][i + 1] = max(probs)
                backpointer_matrix[j][i + 1] = np.argmax(np.array(probs))

        best_path_pointer = np.argmax(viterbi_matrix[:, n - 1])
        tags_for_the_sentence = []
        best_decoded_path = []
        for k in range(n - 1, -1, -1):
            best_decoded_path.append(best_path_pointer)
            best_path_pointer = backpointer_matrix[best_path_pointer][k]
        best_decoded_path = reversed(best_decoded_path)
        for tag_ind in best_decoded_path:
            tags_for_the_sentence.append(unique_tags[tag_ind])
        return tags_for_the_sentence

    predicted_tags_viterbi = []
    for sentence in data:
        predicted_tags_viterbi.append(viterbi_one_sentence(sentence, unique_tags))

    return predicted_tags_viterbi


def accuracy(ground_truth_tags, predicted_tags):
    return sum(
        [a == b for a, b in zip(ground_truth_tags, [word for sublist in predicted_tags for word in sublist])]) / len(
        ground_truth_tags)


def heuristic_learn(train_data):
    unambiguous_tags = dict()
    for i, sent in enumerate(train_data):
        for j, word in enumerate(sent):
            if word[1] in unambiguous_tags and unambiguous_tags[word[1]] != word[2]:
                unambiguous_tags[word[1]] = '<ambiguous>'
                continue
            unambiguous_tags[word[1]] = word[2]
    return unambiguous_tags


def heuristic_apply(data, unambiguous_tags, predicted_tags):
    for i, sent in enumerate(data):
        for j, word in enumerate(sent):
            if word[1] in unambiguous_tags and unambiguous_tags[word[1]] != '<ambiguous>':
                predicted_tags[i][j] = unambiguous_tags[word[1]]


def main():
    word_counter_final, train_data = create_vocabulary()
    tag_counter, emission_probabilities, transition_probabilities = model_learning(word_counter_final, train_data)

    with open('data/dev') as file:
        s = file.read().splitlines()
        dev_data = []
        temp = []
        ground_truth_tags = []
        for d in s:
            if d != '':
                word_desc = d.split('\t')
                temp.append(word_desc)
                ground_truth_tags.append(word_desc[2])
            else:
                dev_data.append(temp)
                temp = []
        dev_data.append(temp)

    dev_greedy_tags = greedy_decoding(dev_data, word_counter_final, tag_counter, emission_probabilities,
                                      transition_probabilities)

    print(f"Accuracy dev_data with Greedy Decoding = {accuracy(ground_truth_tags, dev_greedy_tags)}")

    dev_viterbi_tags = viterbi_decoding(dev_data, word_counter_final, tag_counter, emission_probabilities,
                                        transition_probabilities)
    print(f"Accuracy dev_data with Viterbi Decoding = {accuracy(ground_truth_tags, dev_viterbi_tags)}")

    # Print With heuristic
    unambiguous_tags = heuristic_learn(train_data)
    heuristic_apply(dev_data, unambiguous_tags, dev_greedy_tags)
    heuristic_apply(dev_data, unambiguous_tags, dev_viterbi_tags)

    print(f"Accuracy dev_data with Greedy Decoding and Heuristic = {accuracy(ground_truth_tags, dev_greedy_tags)}")
    print(f"Accuracy dev_data with Viterbi Decoding and Heuristic = {accuracy(ground_truth_tags, dev_viterbi_tags)}")

    with open('data/test') as file:
        s = file.read().splitlines()
        test_data = []
        temp = []
        for d in s:
            if d != '':
                word_desc = d.split('\t')
                temp.append(word_desc)
            else:
                test_data.append(temp)
                temp = []
        test_data.append(temp)
    # print(test_data)
    test_greedy_tags = greedy_decoding(test_data, word_counter_final, tag_counter, emission_probabilities,
                                       transition_probabilities)

    heuristic_apply(test_data, unambiguous_tags, test_greedy_tags)

    with open('greedy.out', 'w') as file:
        for i, sentence in enumerate(test_data):
            for j, word_desc in enumerate(sentence):
                write = f"{j + 1}\t{word_desc[1]}\t{test_greedy_tags[i][j]}\n"
                file.write(write)
            file.write('\n')

    test_viterbi_tags = viterbi_decoding(test_data, word_counter_final, tag_counter, emission_probabilities,
                                         transition_probabilities)

    heuristic_apply(test_data, unambiguous_tags, test_viterbi_tags)
    #
    with open('viterbi.out', 'w') as file:
        for i, sentence in enumerate(test_data):
            for j, word_desc in enumerate(sentence):
                write = f"{j + 1}\t{word_desc[1]}\t{test_viterbi_tags[i][j]}\n"
                file.write(write)
            file.write('\n')


if __name__ == '__main__':
    main()
