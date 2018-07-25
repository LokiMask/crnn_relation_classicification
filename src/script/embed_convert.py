import numpy as np

def convert_dep_embedding(dep_embed_file, words_file, embed_npfile):
    vocab = []
    embed = []
    with open(dep_embed_file) as f:
        for line in f:
            ss = line.strip().split()
            if len(ss) < 300:
                continue
            vocab.append(ss[0])
            vec = [float(x) for x in ss[1:]]
            embed.append(vec)
    with open(words_file, 'w') as f:
        for w in vocab:
            f.write('%s\n' % w)

    embed = np.asarray(embed)
    embed = embed.astype(np.float32)
    np.save(embed_npfile, embed)

convert_dep_embedding("./deps.words.300d.txt", "../../data/depwords.lst", "../../data/dep300.npy")
