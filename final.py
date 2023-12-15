import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv('games.csv')


print(data.head())


# Анализ ценовой политики и продаж

sorted_data1 = data.sort_values('Estimated owners', ascending=False)

plt.figure(figsize=(16, 6))

sns.scatterplot(x='Price', y='Estimated owners', data=sorted_data1)
sns.scatterplot(x='Price', y='Estimated owners', data=sorted_data1)
plt.xlabel('Price')
plt.ylabel('Estimated owners')
plt.title('Price vs Estimated Owners')

mean_price = sorted_data1['Price'].median()

plt.axvline(x=mean_price, color='red', linestyle='--', label=f'Median Price: {mean_price:.2f}')
plt.legend()  # Добавление легенды

plt.show()

# Анализ оценок критиков

data_clean = data.dropna(subset=['Metacritic score'])

data_filtered = data_clean.query('`Metacritic score` != 0')

plt.figure(figsize=(8, 6))
sns.violinplot(data=data_filtered[['Metacritic score']], palette='Pastel1')
plt.title('Оценки критиков')
plt.ylabel('Оценка')

plt.show()

# Анализ оценок пользователей

data_clean = data.dropna(subset=['User score'])

data_filtered = data_clean.query('`User score` != 0')

plt.figure(figsize=(8, 6))
sns.violinplot(data=data_filtered[['User score']], palette='Pastel1')
plt.title('Оценка пользователей')
plt.ylabel('Оценка')

plt.show()


# Анализ платформенной поддержки
platform_support = data[['Windows', 'Mac', 'Linux']]
platform_support.sum().plot(kind='bar')
plt.xlabel('Платформа')
plt.ylabel('Кол-во игр поддерживающие данные платформы')
plt.title('Поддерживаемые платформы')
plt.show()


# Анализ жанров

genre_counts = data['Genres'].value_counts()

top_n = 20
top_genres = genre_counts.head(top_n)


plt.figure(figsize=(8, 10))
top_genres.sort_values().plot(kind='barh', color='skyblue')
plt.ylabel('Жанр')
plt.xlabel('Кол-во игр с данными жанрами')
plt.title(f'Топ {top_n} жанраов с самым большим кол-вом проектов')

plt.tight_layout()
plt.show()

#Топ игр по среднему времени игры за всё время
top_games = data.nlargest(30, 'Average playtime forever')

median_playtime = top_games['Average playtime forever'].median()

plt.figure(figsize=(12, 12))

bar_width = 0.35
index = np.arange(len(top_games))

plt.barh(index, top_games['Average playtime forever'], bar_width, color='skyblue', label='Average playtime forever')

plt.axvline(median_playtime, color='red', linestyle='--', label=f'Median: {median_playtime:.2f} minutes')

plt.xlabel('Время игры (в минутах)')
plt.ylabel('Игра')
plt.title('Топ игр по среднему времени игры за всё время')
plt.yticks(index + bar_width / 2, top_games['Name'])
plt.legend()
plt.tight_layout()

plt.show()

#Распределение отзывов по n жанрам с наибольшим кол-во игр
data['Genres'] = data['Genres'].astype(str)

data['Genres'] = data['Genres'].str.split(', ')
data['Genres'] = data['Genres'].apply(lambda x: x[0] if isinstance(x, list) and x else 'Unknown')  # Оставляем только первый жанр

data_with_reviews = data[(data['Positive'] != 0) | (data['Negative'] != 0)]

top_n = 20
top_genres = data_with_reviews['Genres'].value_counts().head(top_n).index.tolist()

filtered_data = data_with_reviews[data_with_reviews['Genres'].isin(top_genres)]

genre_reviews = filtered_data.groupby('Genres')[['Positive', 'Negative']].sum()

median_value = genre_reviews.median().sum()

plt.figure(figsize=(10, 8))

genre_reviews.plot(kind='bar', stacked=True, color=['lightgreen', 'lightcoral'])
plt.xlabel('Жанр')
plt.ylabel('Количество отзывов')
plt.ylim(0, 12000000)
plt.axhline(y=median_value, color='blue', linestyle='--', label=f'Median: {median_value}')  # Добавление горизонтальной линии для медианы
plt.title(f'Распределение отзывов по {top_n} жанрам с наибольшим кол-во игр')

plt.legend(loc='upper right')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


#Сравнение количества сетевых и несетевых игр для N самых популярных жанров
# Функция для определения сетевой игры
def is_multiplayer(description):
    if pd.notnull(description) and ('Multi-player' in description.split(',')):
        return True
    else:
        return False

data['Multi-player'] = data['Categories'].apply(is_multiplayer)

genre_counts = data['Genres'].value_counts()

top_N =20
top_genres = genre_counts.head(top_N).index.tolist()

filtered_data = data[data['Genres'].isin(top_genres)]

genre_multiplayer_count = filtered_data.groupby(['Genres', 'Multi-player']).size().unstack(fill_value=0)

genre_multiplayer_count.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title(f'Сравнение количества сетевых и несетевых игр для {top_N} самых популярных жанров')
plt.xlabel('Жанры')
plt.ylabel('Количество игр')
plt.legend(labels=['Несетевая', 'Сетевая'])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#Топ жанров сетевых игр
data['Multi-player'] = data['Categories'].apply(is_multiplayer)

multiplayer_games = data[data['Multi-player'] == True]

genre_counts = multiplayer_games['Genres'].value_counts()

top_N = 20
top_genres = genre_counts.head(top_N)

top_genres.plot(kind='bar', figsize=(10, 6))
plt.title('Топ жанров сетевых игр')
plt.xlabel('Жанры')
plt.ylabel('Количество игр')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#Доля сетевых игр для N самых популярных жанров
data['Multi-player'] = data['Categories'].apply(is_multiplayer)

multiplayer_games = data[data['Multi-player'] == True]

genre_counts = multiplayer_games['Genres'].value_counts()

top_N = 20
top_genres = genre_counts.head(top_N)

genre_share = top_genres / top_genres.sum()

plt.figure(figsize=(8, 8))
plt.pie(genre_share, labels=genre_share.index, autopct='%1.1f%%', startangle=140)
plt.title(f'Доля сетевых игр для {top_N} самых популярных жанров')
plt.axis('equal')
plt.tight_layout()
plt.show()