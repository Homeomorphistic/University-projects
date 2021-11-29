DROP TABLE skoki;
DROP TABLE Serie;
DROP TABLE Zawody;
DROP TABLE Skocznie;
DROP TABLE Typy_skoczni;
DROP TABLE Zawodnicy;
-----------------------------------------------------------
--Tabela ze wszystkimi statystykami skoków z wielu zawodów
CREATE TABLE Skoki (
	nazwisko VARCHAR,
	imie VARCHAR,
	kraj VARCHAR,
	predkosc DECIMAL(3,1),
	dystans DECIMAL(4,1),
	punkty_za_dystans DECIMAL(4,1),
	sedzia1 DECIMAL(3,1),
	sedzia2 DECIMAL(3,1),
	sedzia3 DECIMAL(3,1),
	sedzia4 DECIMAL(3,1),
	sedzia5 DECIMAL(3,1),
	punkty_sedzia DECIMAL(4,1),
	belka INTEGER,
	punkty_belka DECIMAL(3,1),
	wiatr DECIMAL(3,2),
	punkty_wiatr DECIMAL(3,1),
	ogolnie_punkty DECIMAL(4,1),
	miejsce_w_serii INTEGER,
	seria INTEGER,
	data VARCHAR,
	skocznia VARCHAR
);

\copy skoki FROM 'Skoki.csv' NULL 'NA' DELIMITER ';';
----------------------------------------------------------------

CREATE TABLE Zawodnicy (
	id_zawodnika SERIAL PRIMARY KEY,
	nazwisko VARCHAR NOT NULL,
	imie VARCHAR NOT NULL,
	kraj VARCHAR NOT NULL CHECK (length(kraj) = 3)
);

INSERT INTO Zawodnicy(nazwisko, imie, kraj)
	SELECT DISTINCT nazwisko, imie, kraj
	FROM Skoki;
---------------------------------------------------------------

CREATE TABLE Typy_skoczni (
	id_typu SERIAL PRIMARY KEY,
	nazwa_typu VARCHAR NOT NULL,
	przelicznik_dystansu DECIMAL(3,1) NOT NULL,
	minimum INTEGER,
	maksimum INTEGER
);

\copy Typy_skoczni(nazwa_typu, przelicznik_dystansu, minimum, maksimum) FROM 'Typy_skoczni.csv' NULL 'NA' DELIMITER ';'; --CSV HEADER;
---------------------------------------------------------------

CREATE TABLE  Skocznie (
	id_skoczni SERIAL PRIMARY KEY,
	kraj VARCHAR NOT NULL,
	miasto VARCHAR NOT NULL,
	id_typu INTEGER REFERENCES Typy_skoczni(id_typu) 
		ON UPDATE CASCADE ON DELETE RESTRICT NOT NULL,
	punkt_K DECIMAL(4,1) NOT NULL CHECK (punkt_K>0) 
);
--Przed skopiowaniem należy stworzyć wyzwalacz typ_trigger, który znajduje się na końcu pliku
\copy Skocznie(miasto, kraj, punkt_K) FROM 'Skocznie.csv' NULL 'NA' DELIMITER ';' CSV HEADER;

--INSERT INTO Skocznie(miasto, kraj, id_typu)
--	SELECT DISTINCT left(skocznia, position(' ' IN skocznia)-1), 
--			substring(skocznia, length(skocznia)-3, 3),
--		 	2
--	FROM skoki;
------------------------------------------------------------------------

CREATE TABLE Zawody (
	id_zawodow SERIAL PRIMARY KEY,
	nazwa VARCHAR,
	id_skoczni INTEGER REFERENCES Skocznie(id_skoczni) ON UPDATE CASCADE ON DELETE RESTRICT NOT NULL,
	data DATE UNIQUE NOT NULL
);

INSERT INTO Zawody(nazwa, id_skoczni, data)
	SELECT DISTINCT 'Puchar swiata', id_skoczni, to_date(data, 'DD Mon YYYY') AS d
	FROM Skocznie JOIN skoki ON(Skocznie.miasto = left(skocznia, length(skocznia)-6));
--------------------------------------------------------------------------

CREATE TABLE Serie (
	id_serii SERIAL PRIMARY KEY,
	id_zawodnika INTEGER REFERENCES Zawodnicy(id_zawodnika) ON UPDATE CASCADE ON DELETE RESTRICT NOT NULL,
	id_zawodow INTEGER REFERENCES Zawody(id_zawodow) ON UPDATE CASCADE ON DELETE RESTRICT NOT NULL,
	nr_serii INTEGER NOT NULL, --1 lub 2, oznaczający numer skoku
	predkosc DECIMAL(3,1) NOT NULL CHECK (predkosc>=0),
	dystans DECIMAL(4,1) NOT NULL CHECK (dystans>=0), 
	sedzia1 DECIMAL(3,1) NOT NULL CHECK (sedzia1>=0 AND sedzia1<=20),
	sedzia2 DECIMAL(3,1) NOT NULL CHECK (sedzia2>=0 AND sedzia2<=20),
	sedzia3 DECIMAL(3,1) NOT NULL CHECK (sedzia3>=0 AND sedzia3<=20),
	sedzia4 DECIMAL(3,1) NOT NULL CHECK (sedzia4>=0 AND sedzia4<=20),
	sedzia5 DECIMAL(3,1) NOT NULL CHECK (sedzia5>=0 AND sedzia5<=20),
	punkty_belka DECIMAL(3,1) NOT NULL,
	punkty_wiatr DECIMAL(3,1) NOT NULL,
	UNIQUE(id_zawodnika, id_zawodow, nr_serii)
);

INSERT INTO Serie
	SELECT Zawodnicy.id_zawodnika, Zawody.id_zawodow, 
			seria, predkosc, dystans,
			sedzia1, sedzia2, sedzia3, sedzia4, sedzia5, punkty_belka, punkty_wiatr 
	FROM skoki S JOIN Zawody ON (to_date(S.data, 'DD Mon YYYY') = Zawody.data) 
			JOIN Zawodnicy ON (S.nazwisko = Zawodnicy.nazwisko AND S.imie = Zawodnicy.imie);

----------------------------FUNCTIONS-----------------------------------------------
--Funkcja wyliczająca ilość punktów od sędziów
CREATE FUNCTION noty(s1 DECIMAL(3,1), s2 DECIMAL(3,1), s3 DECIMAL(3,1),
 s4 DECIMAL(3,1), s5 DECIMAL(3,1)) RETURNS DECIMAL(4,1) AS 
	$$
		BEGIN
			RETURN s1+s2+s3+s4+s5 - greatest(s1,s2,s3,s4,s5) - least(s1,s2,s3,s4,s5);
		END;
	$$ LANGUAGE 'plpgsql';

--Funkcja wyliczająca punkty za dystans (w zależności od punktu_K i przelicznika)
CREATE FUNCTION punkty_dystans(pd DECIMAL(3,1), d DECIMAL(4,1), pk DECIMAL(4,1)) RETURNS DECIMAL(4,1) AS 
	$$
		BEGIN
			RETURN 60 + pd*(d-pk);
		END;
	$$ LANGUAGE 'plpgsql';

---------------------VIEWS-------------------------------------------------
--Widok pomocniczy
CREATE VIEW zawody_skocznie AS (
	SELECT id_skoczni, id_zawodow, data, kraj, miasto, punkt_K, przelicznik_dystansu
	FROM Zawody JOIN Skocznie USING(id_skoczni) JOIN Typy_skoczni USING(id_typu)
);

--Widok z rankingiem z zawodów z Wisły w roku 2018
CREATE VIEW wisla18_suma AS (
	SELECT nazwisko, imie,
	sum(noty(sedzia1,sedzia2,sedzia3,sedzia4,sedzia5)+punkty_belka+punkty_wiatr+
		punkty_dystans(przelicznik_dystansu, dystans, punkt_K)) AS Suma
	FROM Serie JOIN zawody_skocznie USING(id_zawodow) JOIN Zawodnicy USING(id_zawodnika)
	WHERE miasto = 'Wisla' AND data = '2018-11-18'
	GROUP BY nazwisko, imie
	ORDER BY Suma DESC
);

-----------------TRIGGERS-------------------------------
--Funkcja poprawiająca odpowiednie stringi (nazwisko i kraj dużymi literami, imię zaczynające z dużej) 
CREATE FUNCTION duze_litery() RETURNS TRIGGER AS 
	$$
		BEGIN
			NEW.nazwisko = upper(NEW.nazwisko);
			NEW.imie = lower(NEW.imie);
			NEW.imie = initcap(NEW.imie);
			NEW.kraj = upper(NEW.kraj);
			RETURN NEW;
		END;
	$$ LANGUAGE 'plpgsql';
CREATE TRIGGER litery_trigger BEFORE INSERT OR UPDATE ON Zawodnicy
FOR EACH ROW EXECUTE PROCEDURE duze_litery();

--Funkcja przydzielająca odpowiedni typ skoczni w zależności od punktu_K
CREATE FUNCTION typ_skoczni() RETURNS TRIGGER AS 
	$$
	DECLARE r RECORD;
		BEGIN
			FOR r IN SELECT * FROM Typy_skoczni LOOP
				IF (NEW.punkt_K>=r.minimum AND (NEW.punkt_K<=r.maksimum OR r.maksimum IS NULL)) THEN
					NEW.id_typu := r.id_typu;
				END IF;
			END LOOP;
			RETURN NEW;
		END;
	$$ LANGUAGE 'plpgsql';
CREATE TRIGGER typ_trigger BEFORE INSERT OR UPDATE ON Skocznie
FOR EACH ROW EXECUTE PROCEDURE typ_skoczni();
