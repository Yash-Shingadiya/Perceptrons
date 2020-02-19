from __future__ import division
import glob
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

#Perceptron after filtering out stop words
def PerceptronWithoutStopWords(iteration,set_of_learning_rates):
    
    #Using SnowballStemmer as given in the documentation of assignment to get root words
    tokenizer = RegexpTokenizer("[a-zA-Z]+");
    #Using nltk to filter out stopwords
    stemmer = SnowballStemmer("english")
    #Using nltk to filter out stopwords
    stop_words = set(stopwords.words("english"))

    for learning_rate in set_of_learning_rates:
        bag_of_words = {}
        #Reading all the files inside train folder and gathering all the words to form the vocabulary
        path_of_all_training_files = glob.glob("train/*/*" + ".txt")
        for file in path_of_all_training_files:
            with open(file,'r') as fp:
                #Reading each line of the file
                for line in fp:
                    #tokenizing
                    tokens = tokenizer.tokenize(line)
                    #Stemming the files by getting only the unique words means ignoring the plural forms and selecting vocab language as english
                    stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                    #Once words are stemmed, selecting only the unique words from entire bag of words
                    for word in stemmed_words:
                        if word != '':
                            if word.lower() in bag_of_words:
                                bag_of_words[word.lower()] += 1
                            else:
                                bag_of_words[word.lower()] = 1      
            
        for i in bag_of_words:
            #Initializing weight of each word to zero
            bag_of_words[i] = 0.0
        
        path_of_all_ham_files = glob.glob("train/ham/*" + ".txt")
        path_of_all_spam_files = glob.glob("train/spam/*" + ".txt")

        #Updating weights    
        for each in range(iteration):
            #Assume ham as negative class, so target = 0     
            updated_weights_for_ham = update_weights(path_of_all_ham_files, 0.0, bag_of_words, learning_rate, tokenizer, stemmer, stop_words)
            #Assume spam as positive class, so target = 1
            updated_weights_for_spam = update_weights(path_of_all_spam_files, 1.0, bag_of_words, learning_rate, tokenizer, stemmer, stop_words)
       
        #Once the preprocessing of files is done and the weights are updated, calculating the accuracy of the model during training  
        #Calculating number of correct guesses in train/ham during training and determining the accuracy                      
        training_file_count = 0
        #For ham
        training_target = 0
        number_of_correct_guesses_during_training = 0
        #Calculating the total number of words in training folder
        training_file_count = training_file_count + len(path_of_all_ham_files);
        #Reading all the ham files inside train/ham folder and gathering all the ham words
        for file in path_of_all_ham_files:
            bag_of_words = {}
            #Reading each line in the file
            for file in glob.glob(file):
                with open(file,'r') as fp:
                    for line in fp:
                        #tokenizing
                        tokens = tokenizer.tokenize(line)
                        #stemming the files by getting only the unique words means ignoring the plural forms and selecting vocab language as english
                        stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                        #Once words are stemmed, selecting only the unique words from entire bag of words
                        for word in stemmed_words:
                            if word != '':
                                if word.lower() in bag_of_words:
                                    bag_of_words[word.lower()] += 1
                                else:
                                    bag_of_words[word.lower()] = 1      

            updated_weights = 1.0;
            for word in bag_of_words:
                #If word does not have any weight because it was previously not there in bag of words then initialize it to zero
                if word not in updated_weights_for_ham:
                    updated_weights_for_ham[word] = 0.0
                #Finding updated_weights for each word and updating the weights inside train/ham
                activation = (updated_weights_for_ham[word] * bag_of_words[word])
                updated_weights = updated_weights + activation
            
            #If updated_weights is greater than zero then positive class and here the positive class is assumed as spam class
            if updated_weights > 0:
                output_class = 1.0
            #If updated_weights is less than zero then negative class and here the negative class is assumed as ham class
            else:
                output_class = 0.0
            
            if (output_class == training_target):
                number_of_correct_guesses_during_training = number_of_correct_guesses_during_training + 1
        
        #For spam        
        training_target = 1
        training_file_count = training_file_count + len(path_of_all_spam_files);
        #Reading all the spam files inside train/spam folder and gathering all the spam words
        for file in path_of_all_spam_files:
            bag_of_words = {}
            for file in glob.glob(file):
                with open(file,'r') as fp:
                    #Reading each line in the file
                    for line in fp:
                        #tokenizing
                        tokens = tokenizer.tokenize(line)
                        #stemming the files by getting only the unique words means ignoring the plural forms and selecting vocab language as english
                        stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                        #Once words are stemmed, selecting only the unique words from entire bag of words
                        for word in stemmed_words:
                            if word != '':
                                if word.lower() in bag_of_words:
                                    bag_of_words[word.lower()] += 1
                                else:
                                    bag_of_words[word.lower()] = 1      
            
            updated_weights = 1.0;
            #If word does not have any weight because it was previously not there in bag of words then initialize it to zero
            for word in bag_of_words:
                if word not in updated_weights_for_spam:
                    updated_weights_for_spam[word] = 0.0
                
                #Finding updated_weights for each word and updating the weights inside train/spam
                activation = (updated_weights_for_spam[word] * bag_of_words[word])
                updated_weights = updated_weights + activation
            
            #If updated_weights is greater than zero then positive class and here the positive class is assumed as spam class
            if updated_weights > 0:
                output_class = 1.0
            #If updated_weights is less than zero then negative class and here the negative class is assumed as ham class
            else:
                output_class = 0.0
            
            if output_class == training_target :
                number_of_correct_guesses_during_training = number_of_correct_guesses_during_training + 1
        
        training_accuracy = 0
        print "\nFor iteration := ",iteration," and learning_rate := ",learning_rate
        print "Number of correct guesses during training:\t\t\t%d/%s" % (number_of_correct_guesses_during_training, training_file_count)
        training_accuracy = (number_of_correct_guesses_during_training / training_file_count) * 100
        print "Training accuracy := %.2f" % (training_accuracy),"%"

        #Now in similar way, calculating the accuracy of model during test time
        #Calculating number of correct guesses in test/ham during test time and determining the accuracy
        #For ham
        test_file_count = 0
        target = 0
        number_of_correct_guesses = 0
        path_for_ham_test = glob.glob("test/ham/*.txt")
        #Calculating the total number of words in training folder
        test_file_count = test_file_count + len(path_for_ham_test);
        #Reading all the ham files inside test/ham folder and gathering all the ham words
        for file in path_for_ham_test:
            bag_of_words = {}
            for file in glob.glob(file):
                with open(file,'r') as fp:
                    #Reading each line in the file
                    for line in fp:
                        #Tokenizing
                        tokens = tokenizer.tokenize(line)
                        #stemming the files by getting only the unique words means ignoring the plural forms and selecting vocab language as english
                        stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                        #Once words are stemmed, selecting only the unique words from entire bag of words
                        for word in stemmed_words:
                            if word != '':
                                if word.lower() in bag_of_words:
                                    bag_of_words[word.lower()] += 1
                                else:
                                    bag_of_words[word.lower()] = 1      
                
            updated_weights = 1.0;
            #If word does not have any weight because it was previously not there in bag of words then initialize it to zero
            for word in bag_of_words:
                if word not in updated_weights_for_ham:
                    updated_weights_for_ham[word] = 0.0
                
                #Finding updated_weights for each word and updating the weights inside train/ham
                activation = (updated_weights_for_ham[word] * bag_of_words[word])
                updated_weights = updated_weights + activation

            #If updated_weights is greater than zero then positive class and here the positive class is assumed as spam class
            if updated_weights > 0:
                output_class = 1.0
            #If updated_weights is less than zero then negative class and here the negative class is assumed as ham class
            else:
                output_class = 0.0
            
            if (output_class == target):
                number_of_correct_guesses = number_of_correct_guesses + 1    
        
        #For spam
        target = 1
        path_for_spam_test = glob.glob("test/spam/*.txt")
        test_file_count = test_file_count + len(path_for_spam_test);
        #Reading all the spam files inside test/spam folder and gathering all the spam words
        for file in path_for_spam_test:
            bag_of_words = {}
            for file in glob.glob(file):
                with open(file,'r') as fp:
                    #Reading each line in the file
                    for line in fp:
                        #Tokenizing
                        tokens = tokenizer.tokenize(line)
                        #stemming the files by getting only the unique words means ignoring the plural forms and selecting vocab language as english
                        stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                        #Once words are stemmed, selecting only the unique words from entire bag of words
                        for word in stemmed_words:
                            if word != '':
                                if word.lower() in bag_of_words:
                                    bag_of_words[word.lower()] += 1
                                else:
                                    bag_of_words[word.lower()] = 1      
    
            updated_weights = 1.0;
            #If word does not have any weight because it was previously not there in bag of words then initialize it to zero
            for word in bag_of_words:
                if word not in updated_weights_for_spam:
                    updated_weights_for_spam[word] = 0.0

                #Finding updated_weights for each word and updating the weights inside train/ham
                activation = (updated_weights_for_spam[word] * bag_of_words[word])
                updated_weights = updated_weights + activation

            #If updated_weights is greater than zero then positive class and here the positive class is assumed as spam class
            if updated_weights > 0:
                output_class = 1.0
            #If updated_weights is less than zero then negative class and here the negative class is assumed as ham class
            else:
                output_class = 0.0
            
            if (output_class == target):
                number_of_correct_guesses = number_of_correct_guesses + 1    

        test_accuracy = 0
        print "Number of correct guesses during test:  \t\t\t%d/%s" % (number_of_correct_guesses, test_file_count)
        test_accuracy = (number_of_correct_guesses / test_file_count) * 100
        print "Test accuracy := %.2f" % (test_accuracy),"%"


def update_weights(filepath, target, weights, learning_rate, tokenizer, stemmer, stopwords):
    
    for file in filepath:
        bag_of_words = {}
        for file in glob.glob(file):
            with open(file,'r') as fp:
                for line in fp:
                    #Tokenizing
                    tokens = tokenizer.tokenize(line)
                    #stemming the files by getting only the unique words means ignoring the plural forms and selecting vocab language as english
                    stemmed_words = [stemmer.stem(t) for t in tokens if not t in stopwords]
                    #Once words are stemmed, selecting only the unique words from entire bag of words
                    for word in stemmed_words:
                        if word != '':
                            if word.lower() in bag_of_words:
                                bag_of_words[word.lower()] += 1
                            else:
                                bag_of_words[word.lower()] = 1      
        
        updated_weights = 1.0;
        for word in bag_of_words:
            if word not in weights:
                weights[word] = 0.0
            updated_weights += (weights[word]*bag_of_words[word])
        
        #If updated_weights is greater than zero then positive class and here the positive class is assumed as spam class
        if updated_weights > 0:
            output_class = 1.0
        #If updated_weights is less than zero then negative class and here the negative class is assumed as ham class
        else:
            output_class = 0.0
        
        #Updating weights of each word
        for word in bag_of_words:
            delta = (float(learning_rate)*float(target - output_class)*float(bag_of_words[word]))
            weights[word] = weights[word] + delta
    
    return weights

