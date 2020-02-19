Instructions to compile and run:

- To compile and run the program, please type below line on command line: 

python Perceptron.py

- Explanation:
Perceptron.py will call PerceptronWithStopWords(iteration,set_of_learning_rates) and 
PerceptronWithoutStopWords(iteration,set_of_learning_rates) where the accuracies of 
the model will be calculated.

- Note:
Inside the code I have kept iterations = [10,50,100,200] and set_of_learning_rates = [0.001, 0.1, 0.2, 0.5, 0.9].
But, the number of iterations and different values of learning rates can be modified later in the set to run it 
for different values of iterations and learning rates. As when we increase the number of iterations then due to 
lack of computing power the program takes too long to compute the answers.So, I ran the program by setting only 
one value of iterations and keeping 5 different values of learning rates.

Eg: 
iterations = [10] 
set_of_learning_rates = [0.001, 0.1, 0.2, 0.5, 0.9]

iteration = [50]
set_of_learning_rates = [0.001, 0.1, 0.2, 0.5, 0.9]

iteration = [100]
set_of_learning_rates = [0.001, 0.1, 0.2, 0.5, 0.9]

iteration = [200]
set_of_learning_rates = [0.001, 0.1, 0.2, 0.5, 0.9]