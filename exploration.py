import pandas as pd #import the pandas library to read data as a DataFrame object

# loading in our two datasets...we'll combine them later on
# if you load in the data set as I do, they'll need to be placed in the same directory
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# take a peek at the first few rows of both dataframes 
# title, text, subject, data published
true.head()
# load in fake in a separate cell
fake.head()

# Concatenating the dataframes later on, so put a flag in as a feature to both dataframes
# setting entire column to 0 so when we concat the dataframes we know which were real and which were fake
# In this case, the real articles are equal to 0
true["fake_news"] = 0 
fake["fake_news"] = 1 
#print(true)
#print(fake)

# examine the unique article subjects
true["subject"].unique()
fake["subject"].unique()

# overfitting ML model since it would quickly learn that articles that have 
# Reuters in them are true and so the rest aren't
just_text = true["text"]
just_text.head()

# pandas has a handy function called extractall() that accepts a regular expression 
# (regex) pattern as an argument. Regex is a special sequence of characters that 
# defines a search pattern.
# Extract all of the text that comes after the hyphen that follows Reuters.
# We want the strings in the just_text column, and then called extractall() on it.
#  regex	meaning
#   ^	    only at the start of the string
#   .	    any character
#   *	    repeated 0 or more times
#   ?	    make the expression non-greedy
#  ( )	    the text we actually want to capture
# ?P<text>	name the column our text is extracted under to be called "text"
just_text = just_text.str.extractall(r"^.*? - (?P<text>.*)")

# dataframe it created has multiple indices // get rid of match column and move back one col
just_text = just_text.droplevel(1)
#print(just_text)

# reassign just_text to the text column in our true dataframe
true = true.assign(text=just_text["text"]) # switcharoo on the columns

# concat the real and fake dataframes using axis = 0, instead of axis = 1
# so the rows are stack on top of each other.
df = pd.concat([fake, true], axis = 0)

# we won't use these columns in our model we only need the text, we 
# drop them from axis = 1 because we want to remove columns 
df = df.drop(["subject", "date", "title"], axis = 1) 
# check if we have any missing records
df.info()
# we are missing a couple probably due to missing entries, so we 
# use pandas func dropna() on axis = 0 (missing rows) to remove them
df = df.dropna(axis = 0)
# the number of rows should be equal for both columns (text and fake_news)
#df.info()

# save the cleaned csv for other modeling
clean_text = df.to_csv("cleaned_news.csv", index = False) # prevents having two indices