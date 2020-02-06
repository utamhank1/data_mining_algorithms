# This script reads in the netflix dataset (available on
# http://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a) and creates a table with the netflix
# userID as the columns and the movieIDs as the rows.


def main():
    # Import necessary libraries.
    import numpy as np
    import pandas as pd

    # Create empty array.
    rows = []

    # Open the netflix users data set and read every line separating the userID and the movieID by splitting the file
    # by the delimiiter.
    with open("Netflix_data.txt") as file:
        for lines in file.readlines():
            if lines[-2] == ':':
                mID = lines[:-2]
            else:
                row = [mID] + lines.split(',')
                if int(row[2]) > 2:
                    rows.append(row)

    # Create empty pandas dataframe with the desired columns.
    tableDf = pd.DataFrame(data=rows, columns=['Movie_ID', 'User_ID', 'Ratings', 'Date'])

    # Drop the date column and only keep the users that have rated a movie.
    tableDf = tableDf.drop(['Date'], axis=1)
    tableDf['Ratings'] = 1

    # Count the number of movies that a given user has rated.
    ratesCount = tableDf.groupby(['User_ID']).count()['Movie_ID'].reset_index()

    # Only keep the users that have rated 20 movies or more.
    ratesCount = ratesCount[ratesCount['Movie_ID'] <= 20]
    keptUserID = list(ratesCount['User_ID'])
    tableDf = tableDf[tableDf['User_ID'].isin(keptUserID)]

    # Pivot the table to have the userIDs as columns and the rating as rows.
    tableDf = pd.pivot_table(tableDf, values='Ratings', index=['Movie_ID'], columns=['User_ID'], aggfunc=np.sum)

    # Fill all nan values with zeros.
    tableDf = tableDf.fillna(value=0)

    # Parse all ratings as interger values.
    tableArray = tableDf.values.astype(int)

    # Save cleaned dataset to disk.
    np.save('table.npy', tableArray)


if __name__ == "__main__":
    main()
