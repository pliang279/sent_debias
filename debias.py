from __future__ import division, print_function
import numpy as np 
import h5py

word_pairs = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"]]

female_sents = ['A daughter is a person.', 'A female is a person.', 'A gal is a thing.', 'A girl is a person.', 'A herself is a thing.', 'A mother is a person.', 'A woman is a person.', 'Daughters are people.', 'Females are people.', 'Gals are things.', 'Girls are people.', 'Here is Mary.', 'Here is a daughter.', 'Here is a female.', 'Here is a gal.', 'Here is a girl.', 'Here is a herself.', 'Here is a mother.', 'Here is a woman.', 'Here she is.', 'Herselfs are things.', 'It is a gal.', 'It is a herself.', 'It is her.', 'Mary is a person.', 'Mary is here.', 'Mary is there.', 'Mothers are people.', 'She is a person.', 'She is here.', 'She is there.', 'That is Mary.', 'That is a daughter.', 'That is a female.', 'That is a gal.', 'That is a girl.', 'That is a herself.', 'That is a mother.', 'That is a woman.', 'That is her.', 'The daughter is here.', 'The daughter is there.', 'The daughters are here.', 'The daughters are there.', 'The female is here.', 'The female is there.', 'The females are here.', 'The females are there.', 'The gal is here.', 'The gal is there.', 'The gals are here.', 'The gals are there.', 'The girl is here.', 'The girl is there.', 'The girls are here.', 'The girls are there.', 'The herself is here.', 'The herself is there.', 'The herselfs are here.', 'The herselfs are there.', 'The mother is here.', 'The mother is there.', 'The mothers are here.', 'The mothers are there.', "The person's name is Mary.", 'The woman is here.', 'The woman is there.', 'The women are here.', 'The women are there.', 'There is Mary.', 'There is a daughter.', 'There is a female.', 'There is a gal.', 'There is a girl.', 'There is a herself.', 'There is a mother.', 'There is a woman.', 'There she is.', 'These are daughters.', 'These are females.', 'These are gals.', 'These are girls.', 'These are herselfs.', 'These are mothers.', 'These are women.', 'They are daughters.', 'They are females.', 'They are gals.', 'They are girls.', 'They are herselfs.', 'They are mothers.', 'They are women.', 'This is Mary.', 'This is a daughter.', 'This is a female.', 'This is a gal.', 'This is a girl.', 'This is a herself.', 'This is a mother.', 'This is a woman.', 'This is her.', 'Those are daughters.', 'Those are females.', 'Those are gals.', 'Those are girls.', 'Those are herselfs.', 'Those are mothers.', 'Those are women.', 'Women are people.']

male_sents = ['A boy is a person.', 'A father is a person.', 'A guy is a thing.', 'A himself is a thing.', 'A male is a person.', 'A man is a person.', 'A son is a person.', 'Boys are people.', 'Fathers are people.', 'Guys are things.', 'He is a person.', 'He is here.', 'He is there.', 'Here he is.', 'Here is John.', 'Here is a boy.', 'Here is a father.', 'Here is a guy.', 'Here is a himself.', 'Here is a male.', 'Here is a man.', 'Here is a son.', 'Here is his.', 'Himselfs are things.', 'His is here.', 'His is there.', 'It is a guy.', 'It is a himself.', 'It is his.', 'John is a person.', 'John is here.', 'John is there.', 'Males are people.', 'Men are people.', 'Sons are people.', 'That is John.', 'That is a boy.', 'That is a father.', 'That is a guy.', 'That is a himself.', 'That is a male.', 'That is a man.', 'That is a son.', 'The boy is here.', 'The boy is there.', 'The boys are here.', 'The boys are there.', 'The father is here.', 'The father is there.', 'The fathers are here.', 'The fathers are there.', 'The guy is here.', 'The guy is there.', 'The guys are here.', 'The guys are there.', 'The himself is here.', 'The himself is there.', 'The himselfs are here.', 'The himselfs are there.', 'The male is here.', 'The male is there.', 'The males are here.', 'The males are there.', 'The man is here.', 'The man is there.', 'The men are here.', 'The men are there.', "The person's name is John.", 'The son is here.', 'The son is there.', 'The sons are here.', 'The sons are there.', 'There he is.', 'There is John.', 'There is a boy.', 'There is a father.', 'There is a himself.', 'There is a male.', 'There is a man.', 'There is a son.', 'There is his.', 'These are boys.', 'These are fathers.', 'These are guys.', 'These are himselfs.', 'These are males.', 'These are men.', 'These are sons.', 'They are boys.', 'They are himselfs.', 'They are males.', 'They are men.', 'They are sons.', 'This is John.', 'This is a boy.', 'This is a father.', 'This is a guy.', 'This is a himself.', 'This is a male.', 'This is a man.', 'This is a son.', 'This is his.', 'Those are boys.', 'Those are fathers.', 'Those are guys.', 'Those are himselfs.', 'Those are males.', 'Those are men.', 'Those are sons.']

female_records = [False] * len(female_sents)
male_records = [False] * len(male_sents)

print("pairs={}".format(len(word_pairs)))

def word_in_sent(word, sent):
	sent = sent[:-1] # Get rid of period
	sent = sent.lower()
	tokens = sent.split(" ")
	word = word.lower()
	plural = word + 's'
	return (word in tokens) or (plural in tokens)

count = 0
total_sent_pairs = []
for pair in word_pairs:
	print(pair)
	wordA, wordB = pair[0], pair[1]
	sentA, sentB = [], []
	for i, sent in enumerate(female_sents):
		if (word_in_sent(wordA, sent)): 
			sentA.append(sent)
			female_records[i] = True
	for i, sent in enumerate(male_sents):
		if (word_in_sent(wordB, sent)): 
			sentB.append(sent)
			male_records[i] = True
	print(len(sentA), len(sentB))
	if (len(sentA) != len(sentB)):
		print("Not equal")
		print(sentA)
		print(sentB)
		continue
	sent_pairs = zip(sentA, sentB)
	total_sent_pairs += sent_pairs

	count += len(sentA) + len(sentB)

print("Count={}".format(count))
print(len(female_sents), len(male_sents))


print("Unpicked sentences")
for i, picked in enumerate(female_records):
	if (not picked): print(female_sents[i])

for i, picked in enumerate(male_records):
	if (not picked): print(male_sents[i])


print(len(total_sent_pairs))
for pair in total_sent_pairs:
	print("{},".format(pair))


def_sent_pairs = [
('A woman is a person.', 'A man is a person.'),
('Here is a woman.', 'Here is a man.'),
('That is a woman.', 'That is a man.'),
('The woman is here.', 'The man is here.'),
('The woman is there.', 'The man is there.'),
('There is a woman.', 'There is a man.'),
('This is a woman.', 'This is a man.'),
('The women are here.', "The men are here."),
('The women are there.', "The men are there."),
("These are women.", "These are men."),
("Those are women.", "Those are men."),
("They are women.", "They are men."),
("Women are people.", "Men are people."),


('A girl is a person.', 'A boy is a person.'),
('Girls are people.', 'Boys are people.'),
('Here is a girl.', 'Here is a boy.'),
('That is a girl.', 'That is a boy.'),
('The girl is here.', 'The boy is here.'),
('The girl is there.', 'The boy is there.'),
('The girls are here.', 'The boys are here.'),
('The girls are there.', 'The boys are there.'),
('There is a girl.', 'There is a boy.'),
('These are girls.', 'These are boys.'),
('They are girls.', 'They are boys.'),
('This is a girl.', 'This is a boy.'),
('Those are girls.', 'Those are boys.'),
('Here she is.', 'He is a person.'),
('She is a person.', 'He is here.'),
('She is here.', 'He is there.'),
('She is there.', 'Here he is.'),
('There she is.', 'There he is.'),
('A daughter is a person.', 'A son is a person.'),
('Daughters are people.', 'Here is a son.'),
('Here is a daughter.', 'Sons are people.'),
('That is a daughter.', 'That is a son.'),
('The daughter is here.', 'The son is here.'),
('The daughter is there.', 'The son is there.'),
('The daughters are here.', 'The sons are here.'),
('The daughters are there.', 'The sons are there.'),
('There is a daughter.', 'There is a son.'),
('These are daughters.', 'These are sons.'),
('They are daughters.', 'They are sons.'),
('This is a daughter.', 'This is a son.'),
('Those are daughters.', 'Those are sons.'),
('A female is a person.', 'A male is a person.'),
('Females are people.', 'Males are people.'),
('Here is a female.', 'Here is a male.'),
('That is a female.', 'That is a male.'),
('The female is here.', 'The male is here.'),
('The female is there.', 'The male is there.'),
('The females are here.', 'The males are here.'),
('The females are there.', 'The males are there.'),
('There is a female.', 'There is a male.'),
('These are females.', 'These are males.'),
('They are females.', 'They are males.'),
('This is a female.', 'This is a male.'),
('Those are females.', 'Those are males.'),
('A herself is a thing.', 'A himself is a thing.'),
('Here is a herself.', 'Here is a himself.'),
('Herselfs are things.', 'Himselfs are things.'),
('It is a herself.', 'It is a himself.'),
('That is a herself.', 'That is a himself.'),
('The herself is here.', 'The himself is here.'),
('The herself is there.', 'The himself is there.'),
('The herselfs are here.', 'The himselfs are here.'),
('The herselfs are there.', 'The himselfs are there.'),
('There is a herself.', 'There is a himself.'),
('These are herselfs.', 'These are himselfs.'),
('They are herselfs.', 'They are himselfs.'),
('This is a herself.', 'This is a himself.'),
('Those are herselfs.', 'Those are himselfs.'),
('Here is Mary.', 'Here is John.'),
('Mary is a person.', 'John is a person.'),
('Mary is here.', 'John is here.'),
('Mary is there.', 'John is there.'),
('That is Mary.', 'That is John.'),
("The person's name is Mary.", "The person's name is John."),
('There is Mary.', 'There is John.'),
('This is Mary.', 'This is John.'),

('A gal is a thing.', 'A guy is a thing.'),
('Gals are things.', 'Guys are things.'),
('Here is a gal.', 'Here is a guy.'),
('It is a gal.', 'It is a guy.'),
('That is a gal.', 'That is a guy.'),
('The gal is here.', 'The guy is here.'),
('The gal is there.', 'The guy is there.'),
('The gals are here.', 'The guys are here.'),
('The gals are there.', 'The guys are there.'),
('There is a gal.', 'There is a guy.'),
('These are gals.', 'These are guys.'),
('They are gals.', 'They are gusy'),
('This is a gal.', 'This is a guy.'),
('Those are gals.', 'Those are guys.'),

('A mother is a person.', 'A father is a person.'),
('Here is a mother.', 'Fathers are people.'),
('Mothers are people.', 'Here is a father.'),
('That is a mother.', 'That is a father.'),
('The mother is here.', 'The father is here.'),
('The mother is there.', 'The father is there.'),
('The mothers are here.', 'The fathers are here.'),
('The mothers are there.', 'The fathers are there.'),
('There is a mother.', 'There is a father.'),
('These are mothers.', 'These are fathers.'),
('They are mothers.', 'They are fathers.'),
('This is a mother.', 'This is a father.'),
('Those are mothers.', 'Those are fathers.'),

('It is her.', 'It is his.'),
('That is her.', 'That is his.'),
('This is her.', 'This is his.'),
('Here is hers.', 'Here is his.'),
('Hers is here.', 'His is here.'),
('Hers is there.', 'His is there.')
]

print(len(def_sent_pairs))


