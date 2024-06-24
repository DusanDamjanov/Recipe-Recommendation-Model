import loader as loader
import vectorizer as vectorizer
import naive_bayes as naive

if __name__=="__main__":
    data = loader.load()
    vectorizer.vectorize(data)
    
