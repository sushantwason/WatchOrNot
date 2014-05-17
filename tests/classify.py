'''Testing classification of a sentence to topics'''



from nltk.classify import PositiveNaiveBayesClassifier


various_sentences = ['The President did not comment','I lost the keys','The team won the game','Sara has two kids','The ball went off the court','They had the ball for the whole game','The show is over']
sports_sentences = [ 'The team dominated the game','They lost the ball','The game was intense','The goalkeeper catched the ball','The other team controlled the ball' ]





def features(sentence):
        words = sentence.lower().split()
        return dict(('contains(%s)' % w, True) for w in words)




def main():
        positive_featuresets = list(map(features, sports_sentences))
        unlabeled_featuresets = list(map(features, various_sentences))
        classifier = PositiveNaiveBayesClassifier.train(positive_featuresets,unlabeled_featuresets)
        print classifier.classify(features('My team lost the game'))

if __name__=="__main__":
        main()



