from pathlib import Path

import pandas as pd


csv_path = Path(__file__).with_name("vgchartz-2024.csv")
df = pd.read_csv(csv_path)

print("\n1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
print(f"Строк: {df.shape[0]}, столбцов: {df.shape[1]}")
print("\nПервые пять строк:")
print(df.head().to_string(index=False))

print("\nИнформация о датасете:")
df.info()

print("\nОписательная статистика:")
print(df.describe().T.round(4).to_string())

print("\nКоличество уникальных непустых значений:")
print(df.nunique().to_string())

column_descriptions = {
    "img": "Путь к изображению обложки",
    "title": "Название игры",
    "console": "Игровая платформа",
    "genre": "Жанр игры",
    "publisher": "Издатель",
    "developer": "Разработчик",
    "critic_score": "Оценка критиков Metacritic по 10-балльной шкале",
    "total_sales": "Мировые продажи, млн копий",
    "na_sales": "Продажи в Северной Америке, млн копий",
    "jp_sales": "Продажи в Японии, млн копий",
    "pal_sales": "Продажи в Европе и Африке, млн копий",
    "other_sales": "Продажи в остальных регионах, млн копий",
    "release_date": "Дата выхода игры",
    "last_update": "Дата последнего обновления записи",
}

print("\nОписание столбцов:")
for column in df.columns:
    print(f"{column} — {column_descriptions[column]}")

sales = df["total_sales"].dropna()
print("\nСтатистика мировых продаж, млн копий:")
print(f"Мода: {', '.join(sales.mode().astype(str))}")
print(f"Среднее: {sales.mean():.4f}")
print(f"Медиана: {sales.median():.4f}")

categories = ["title", "console", "genre", "publisher", "developer"]
unique_path = Path(__file__).with_name("unique_values.txt")

with unique_path.open("w", encoding="utf-8") as output:
    for column in categories:
        values = df[column].unique()
        output.write(f"\n{column} ({len(values)} значений с учётом NaN)\n")
        for value in values:
            output.write(f"{value}\n")

print(f"\nВсе значения, полученные через unique(), записаны в {unique_path.name}.")

missing = pd.DataFrame({
    "Пропусков": df.isna().sum(),
    "Доля, %": df.isna().mean() * 100,
}).sort_values("Доля, %", ascending=False)

print("\nПропущенные значения:")
print(missing.round(2).to_string())
print(
    f"Больше всего пропусков в {missing.index[0]} "
    f"({missing.iloc[0]['Доля, %']:.2f}%). "
    "Возможно, для многих игр оценки критиков отсутствуют в источнике."
)

full_duplicates = df.duplicated().sum()
repeated_pairs = df.duplicated(subset=["title", "console"]).sum()

print(f"\nПолных дубликатов: {full_duplicates}")
print(f"Повторов пары название–платформа после первого появления: {repeated_pairs}")

example = df.loc[
    df["title"].eq("Hot Shots Golf") & df["console"].eq("PS"),
    ["title", "console", "release_date", "total_sales"],
]
print(example.to_string(index=False))
print(
    "Одинаковое название может встречаться на разных платформах. "
    "Даже у одинаковой пары игра–платформа могут различаться продажи. "
    "Например, для Hot Shots Golf на PS указаны 2,79 и 2,56 млн копий. "
    "По этим полям нельзя уверенно определить ошибочную запись."
)

clean = df.copy()
clean[categories] = clean[categories].fillna("no data")

numeric_columns = clean.select_dtypes(include="number").columns
numeric_samples = {
    column: clean.dropna(subset=[column])[column]
    for column in numeric_columns
}
sales_data = clean.dropna(subset=["total_sales"]).copy()

print(f"\nРазмер исходного каталога: {df.shape}")
print(f"Размер каталога после обработки категорий: {clean.shape}")
print(f"Размер выборки с известными мировыми продажами: {sales_data.shape}")
print(f"Пропусков в категориях после fillna: {clean[categories].isna().sum().sum()}")
print(
    "Полных дубликатов нет, поэтому по результатам проверки строки не удалялись. "
    "Для каждого числового показателя пропуски исключаются из его выборки. "
    "Подстановка среднего или медианы при высокой доле пропусков "
    "изменила бы распределение. Строка 'no data' в числовом поле "
    "мешала бы выполнять арифметические операции."
)

print("\n2. АСИММЕТРИЯ, ЭКСЦЕСС И КОЭФФИЦИЕНТ ВАРИАЦИИ")

sales = numeric_samples["total_sales"]
mean_sales = sales.mean()
std_sales = sales.std()
skew_sales = sales.skew()
kurt_sales = sales.kurt()

distribution = pd.DataFrame({
    "Среднее, млн": [mean_sales],
    "Медиана, млн": [sales.median()],
    "Стандартное отклонение, млн": [std_sales],
    "Асимметрия": [skew_sales],
    "Эксцесс": [kurt_sales],
}, index=["Мировые продажи"])

print(distribution.round(4).to_string())

cv = std_sales / mean_sales * 100
q1, q3 = sales.quantile([0.25, 0.75])
upper_fence = q3 + 1.5 * (q3 - q1)
high_values = sales.gt(upper_fence).sum()

print(f"Коэффициент вариации: {cv:.2f}%")
print(
    f"Асимметрия {skew_sales:.2f} указывает на длинный правый хвост. "
    f"Эксцесс {kurt_sales:.2f} указывает на сильное влияние крайних значений. "
    f"По правилу 1,5 IQR найдено {high_values} значений "
    f"выше {upper_fence:.3f} млн копий."
)
print(f"Стандартное отклонение в {cv / 100:.2f} раза больше среднего.")

print("\n3. ФИЛЬТРАЦИЯ, СОРТИРОВКА, СРЕЗЫ И ГРУППИРОВКА")

million_sellers = sales_data.query("total_sales >= 1")
print(f"\nquery() отобрал {len(million_sellers)} записей с продажами от 1 млн.")
print(million_sellers[["title", "console", "total_sales"]].head(10).to_string(index=False))

by_score = clean.dropna(subset=["critic_score"]).sort_values(
    by="critic_score", ascending=False, kind="stable"
)
print(f"\nsort_values() отсортировал {len(by_score)} записей по убыванию оценки.")
print(by_score[["title", "console", "critic_score"]].head(10).to_string(index=False))

print("\nloc — строки с метками от 0 до 4 и выбранные по имени столбцы:")
print(clean.loc[0:4, ["title", "console", "total_sales"]].to_string(index=False))

ps4_releases = clean.loc[
    clean["console"].eq("PS4"), ["title", "genre", "release_date"]
]
print(f"\nВыборка по условию — {len(ps4_releases)} записей для PS4. Первые 10:")
print(ps4_releases.head(10).to_string(index=False))

print("\niloc — первые пять строк и столбцы с позициями 1, 2, 3:")
print(clean.iloc[:5, 1:4].to_string(index=False))

genre_sales = sales_data.groupby("genre")["total_sales"].agg(
    mean_sales="mean",
    records="count",
).sort_values("mean_sales", ascending=False)

print("\nСредние мировые продажи по жанрам и количество учтённых записей:")
print(genre_sales.round(3).to_string())
print(
    "У Sandbox среднее составляет 1,89 млн, но оно рассчитано по одной записи. "
    "У Shooter среднее около 0,67 млн по 1 485 записям."
)

print("\n4. ТОП-5 ЗАПИСЕЙ ПО МИРОВЫМ ПРОДАЖАМ")

top5 = sales_data.nlargest(5, "total_sales")[
    ["title", "console", "genre", "publisher", "total_sales"]
].reset_index(drop=True)
top5.index = pd.RangeIndex(1, len(top5) + 1, name="Место")

print(top5.to_string())
print(
    "Мировые продажи показывают коммерческий масштаб игры в количестве копий. "
    "Рейтинг составлен по отдельным записям для каждой платформы. "
    "На позиции могли повлиять известность серии, популярность платформы "
    "и время, за которое накопились продажи."
)