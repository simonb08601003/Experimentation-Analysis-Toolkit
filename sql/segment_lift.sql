-- Visit-rate lift by channel and tenure, biggest first.
WITH by_cell AS (
    SELECT
        channel,
        CASE WHEN newbie = 1 THEN 'new' ELSE 'returning' END     AS tenure,
        segment,
        COUNT(*)    AS users,
        AVG(visit)  AS visit_rate
    FROM users
    GROUP BY ALL
),
with_control AS (
    SELECT
        c.*,
        MAX(CASE WHEN segment = 'No E-Mail' THEN visit_rate END)
            OVER (PARTITION BY channel, tenure)                    AS control_rate
    FROM by_cell c
)
SELECT
    channel, tenure, segment AS arm, users,
    ROUND(visit_rate, 4)                       AS visit_rate,
    ROUND(control_rate, 4)                     AS control_rate,
    ROUND(visit_rate - control_rate, 4)        AS abs_lift,
    RANK() OVER (ORDER BY visit_rate - control_rate DESC) AS lift_rank
FROM with_control
WHERE segment <> 'No E-Mail'
ORDER BY lift_rank;
