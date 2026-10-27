-- Spend lift by decile of last year's spend.
WITH ranked AS (
    SELECT *, NTILE(10) OVER (ORDER BY history) AS history_decile
    FROM users
)
SELECT
    history_decile,
    ROUND(MIN(history), 1) AS history_min,
    ROUND(MAX(history), 1) AS history_max,
    AVG(CASE WHEN segment = 'Mens E-Mail'   THEN spend END) - AVG(CASE WHEN segment = 'No E-Mail' THEN spend END) AS mens_spend_lift,
    AVG(CASE WHEN segment = 'Womens E-Mail' THEN spend END) - AVG(CASE WHEN segment = 'No E-Mail' THEN spend END) AS womens_spend_lift
FROM ranked
GROUP BY history_decile
ORDER BY history_decile;
