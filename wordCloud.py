from utils import *
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

def saveWordCloud(df, category):
    # Create and generate a word cloud image:
    print('Creating a wordcloud for category', category)
    data = getCategoryContents(df, category)
    wordcloud=WordCloud(max_font_size=50, max_words=100, background_color="white").generate(data)

    # Save the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(category+".png", format="png")

df = read_csv("train_set.csv")
categories = df.Category.unique()
for category in categories:
    saveWordCloud(df, category)
