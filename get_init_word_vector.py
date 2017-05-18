vector_file = "../model_word2vec/wiki.zh.sogou.vector"
word_file = "../data/word_dict.txt"
out_file = "../data/vectors_word.txt"

def get_vectors():
    vectors = {}
    with open(vector_file) as f:
        for line in f:
            line = line.strip().split()
            if len(line) <= 0:
                continue

            word = line[0]
            vector = " ".join(line[1:])
            vectors[word] = vector

    words = vectors.keys()
    result = {}
    with open(word_file) as f, open(out_file, 'w') as f_out:
        for word in f:
            word = word.strip()
            if word in words:
                result[word] = vectors[word]
            else:
                result[word] = " ".join([str(0.01)] * 300)

        for w,v in result.items():
            f_out.write(v + "\n")

get_vectors()