from PerceptronWithStopWords import PerceptronWithStopWords
from PerceptronWithoutStopWords import PerceptronWithoutStopWords

iterations = [10, 50, 100, 200]
set_of_learning_rates = [0.001, 0.1, 0.2, 0.5, 0.9]

#Calaculates accuracy of Perceptron with stop words
print "\n----------------------------"
print "Preceptron With Stopwords:"
print "----------------------------"
for iteration in iterations:
	PerceptronWithStopWords(iteration,set_of_learning_rates)

#Calaculates accuracy of Perceptron without stop words
print "\n-------------------------------"
print "Perceptron Without Stopwords:"
print "-------------------------------"
for iteration in iterations:
	PerceptronWithoutStopWords(iteration,set_of_learning_rates)
