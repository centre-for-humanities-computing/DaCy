import time

import spacy

x = 30

for n in [1000]:
    for mdl in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]:
        nlp = spacy.load(mdl)
        s = time.time()
        docs = list(nlp.pipe(["Dette er en hurtig og let sætning at læse "* x] * n))
        e = time.time() - s
        print(f"Model {mdl} on {n}  with {x} times the input took {round(e, 2)}")


# Model da_core_news_sm on 10 took 0.01450204849243164
# Model da_core_news_md on 10 took 0.015280008316040039
# Model da_core_news_lg on 10 took 0.014426946640014648
# Model da_core_news_sm on 100 took 0.08396005630493164
# Model da_core_news_md on 100 took 0.09811711311340332
# Model da_core_news_lg on 100 took 0.09838008880615234
# Model da_core_news_sm on 1000 took 0.7954728603363037
# Model da_core_news_md on 1000 took 0.9309251308441162
# Model da_core_news_lg on 1000 took 0.9219119548797607
# Model da_core_news_sm on 10000 took 7.643948078155518
# Model da_core_news_md on 10000 took 8.55750823020935
# Model da_core_news_lg on 10000 took 8.589558124542236
# Model da_core_news_sm on 100000 took 76.37360978126526
# Model da_core_news_md on 100000 took 86.1767578125
# Model da_core_news_lg on 100000 took 86.16969013214111

# Model en_core_web_sm on 10 took 0.3579418659210205
# Model en_core_web_md on 10 took 0.016063928604125977
# Model en_core_web_lg on 10 took 0.02293705940246582
# Model en_core_web_sm on 100 took 0.09960103034973145
# Model en_core_web_md on 100 took 0.10839295387268066
# Model en_core_web_lg on 100 took 0.11303281784057617
# Model en_core_web_sm on 1000 took 0.9938411712646484
# Model en_core_web_md on 1000 took 1.5511338710784912
# Model en_core_web_lg on 1000 took 1.0833830833435059
# Model en_core_web_sm on 10000 took 9.478515863418579
# Model en_core_web_md on 10000 took 11.972906112670898
# Model en_core_web_lg on 10000 took 10.573781967163086