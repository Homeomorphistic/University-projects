-----------PROCEDURES-----------
DELIMITER $$

CREATE FUNCTION climate_diff_to_int(str VARCHAR(50)) RETURNS DECIMAL(2,1)
DETERMINISTIC
BEGIN
    DECLARE diff DECIMAL(2,1);
    IF str = "0T" OR str = "0N" THEN
        SET diff = 0;
    ELSE
        SET diff = CONVERT(str, DECIMAL(2,1));
    END IF;
    RETURN (diff);
END $$

CREATE FUNCTION last_team(name VARCHAR(50)) RETURNS VARCHAR(50)
DETERMINISTIC
BEGIN
    DECLARE last_team VARCHAR(50);
    SELECT team INTO last_team
    FROM players_view
    WHERE player_name = name
    ORDER BY last_date DESC
    LIMIT 1;
    RETURN last_team;
END $$

CREATE FUNCTION most_freq_nat(name VARCHAR(50)) RETURNS VARCHAR(50)
DETERMINISTIC
BEGIN
    DECLARE freq_nation VARCHAR(50);
        SELECT nat INTO freq_nation
        FROM players_view
        WHERE player_name = name
        GROUP BY player_name, nat
        ORDER BY SUM(occur) DESC
        LIMIT 1;
    RETURN freq_nation;
END $$

CREATE FUNCTION most_freq_type(name VARCHAR(50)) RETURNS VARCHAR(50)
DETERMINISTIC
BEGIN
    DECLARE freq_type VARCHAR(50);
        SELECT type INTO freq_type
        FROM players_view
        WHERE player_name = name
        GROUP BY player_name, type
        ORDER BY SUM(occur) DESC
        LIMIT 1;
    RETURN freq_type;
END $$

CREATE FUNCTION most_freq_pos(name VARCHAR(50)) RETURNS VARCHAR(50)
DETERMINISTIC
BEGIN
    DECLARE freq_pos VARCHAR(50);
        SELECT pos INTO freq_pos
        FROM players_view
        WHERE player_name = name
        GROUP BY player_name, pos
        ORDER BY SUM(occur) DESC
        LIMIT 1;
    RETURN freq_pos;
END $$

CREATE FUNCTION most_freq_age(name VARCHAR(50)) RETURNS VARCHAR(50)
DETERMINISTIC
BEGIN
    DECLARE freq_age VARCHAR(50);
        SELECT age INTO freq_age
        FROM players_view
        WHERE player_name = name
        GROUP BY player_name, age
        ORDER BY SUM(occur) DESC
        LIMIT 1;
    RETURN freq_age;
END $$

CREATE FUNCTION most_freq_height(name VARCHAR(50)) RETURNS VARCHAR(50)
DETERMINISTIC
BEGIN
    DECLARE freq_height VARCHAR(50);
        SELECT h INTO freq_height
        FROM players_view
        WHERE player_name = name
        GROUP BY player_name, h
        ORDER BY SUM(occur) DESC
        LIMIT 1;
    RETURN freq_height;
END $$

CREATE FUNCTION most_freq_weight(name VARCHAR(50)) RETURNS VARCHAR(50)
DETERMINISTIC
BEGIN
    DECLARE freq_weight VARCHAR(50);
        SELECT w INTO freq_weight
        FROM players_view
        WHERE player_name = name
        GROUP BY player_name, w
        ORDER BY SUM(occur) DESC
        LIMIT 1;
    RETURN freq_weight;
END $$

DELIMITER ;

--------------TRIGGERS------------
DELIMITER $$

CREATE TRIGGER uc_nations BEFORE INSERT ON Nations
  FOR EACH ROW
  BEGIN
    SET NEW.nation := UPPER(NEW.nation);
    SET NEW.polish_sub := UPPER(NEW.polish_sub);
  END$$

CREATE TRIGGER uc_teams BEFORE INSERT ON Teams
  FOR EACH ROW
  BEGIN
    SET NEW.name := UPPER(NEW.name);
    SET NEW.conference := UPPER(NEW.conference);
    SET NEW.climate := UPPER(NEW.climate);
  END$$

CREATE TRIGGER uc_climate_differences BEFORE INSERT ON Climate_differences
  FOR EACH ROW
  BEGIN
    SET NEW.climate_1 := UPPER(NEW.climate_1);
    SET NEW.climate_2 := UPPER(NEW.climate_2);
  END$$

CREATE TRIGGER uc_player_positions BEFORE INSERT ON Player_positions
  FOR EACH ROW
  BEGIN
    SET NEW.position := UPPER(NEW.position);
    SET NEW.position_eng := UPPER(NEW.position_eng);
    SET NEW.position_pol := UPPER(NEW.position_pol);
    SET NEW.type := UPPER(NEW.type);
  END$$

CREATE TRIGGER uc_players BEFORE INSERT ON Players
  FOR EACH ROW
  BEGIN
    SET NEW.name := UPPER(NEW.name);
    SET NEW.nationality := UPPER(NEW.nationality);
    SET NEW.position := UPPER(NEW.position);
  END$$

CREATE TRIGGER uc_matches BEFORE INSERT ON Matches
  FOR EACH ROW
  BEGIN
    SET NEW.conditions := UPPER(NEW.conditions);
  END$$

DELIMITER ;

--SHOW ALL FUNCTIONS--
SELECT 
    routine_name
FROM
    information_schema.routines
WHERE
    routine_type = 'FUNCTION'
        AND routine_schema = 'mls';