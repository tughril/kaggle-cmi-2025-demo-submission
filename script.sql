SELECT * FROM 'data/train.csv' limit 1;

SELECT sequence_id, count(*) FROM 'data/train.csv' GROUP BY sequence_id ORDER BY count(*);

SELECT sequence_id, count(*), gesture FROM 'data/train.csv' GROUP BY sequence_id, gesture ORDER BY count(*);
