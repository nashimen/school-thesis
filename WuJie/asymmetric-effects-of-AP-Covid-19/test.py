import os, pyltp

from pyltp import SentenceSplitter

sentences = SentenceSplitter.split("早餐很一般，酒店比较老了，是港资企业，插座必须要用转换器才能用，价格的话是周边酒店最便宜的一家了，我们是去办美签，交通比较方便，打车十分钟")

print("\n".join(sentences))

