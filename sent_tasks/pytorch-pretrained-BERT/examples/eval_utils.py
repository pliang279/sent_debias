from pattern.en import pluralize, singularize


def my_pluralize(word):
	if (word in ["he", "she", "her", "hers"]): return word
	if (word == "brother"): return "brothers"
	if (word == "drama"): return "dramas"
	return pluralize(word)

def my_singularize(word):
	if (word in ["hers", "his", "theirs"]): return word
	return singularize(word)


def isInSet(word, word_set):
	for wi in [word, my_pluralize(word), my_singularize(word)]:
		if (wi in word_set): return True
		if (wi.lower() in word_set): return True
	return False