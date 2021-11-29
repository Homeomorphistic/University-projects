-----------------------------------------------------------------------------------------------------------
---------------------------------------------FIXING RDB--------------------------------------------------

--CLIMATE DIFFERENCES--
INSERT INTO Climate_differences VALUES
('AM','BSK',2.5,2),
('AM','CFB',4.0,3);

--PLAYERS--
UPDATE Players 
SET position = 'RB'
WHERE name = 'SHAFT BREWER';

UPDATE Players 
SET position = 'LM'
WHERE name = 'PETER-LEE VASSELL';

ALTER TABLE Players MODIFY position VARCHAR(3) NOT NULL;