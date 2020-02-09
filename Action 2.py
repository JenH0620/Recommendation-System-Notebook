'''import nltk
nltk.download("stopwords")
'''
from nltk.corpus import stopwords
list_stopWords=list(set(stopwords.words('english')))
#stop_words = set( stopwords.words( 'english' ) )
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


def create_word_cloud(f):
    #f = list_stopWords( f )

    cut_text = word_tokenize(f)
    cut_text = [w for w in cut_text if not w in list_stopWords]
    cut_text = " ".join( cut_text )
    wc = WordCloud(
        max_words=100,
        width=2000,
        height=1200,
    )
    wordcloud = wc.generate( cut_text )
    wordcloud.to_file( "wordcloud.jpg" )


data = open( './Market_Basket_Optimisation.csv','r').read()
print(data)
create_word_cloud( data )
#print(stopwords.words( 'english' ))
