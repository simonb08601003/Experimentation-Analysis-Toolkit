-- Scorecard per arm. Revenue per visit is a ratio metric, so its test lives in Python (delta method).
SELECT
    segment                                   AS arm,
    COUNT(*)                                  AS users,
    AVG(visit)                                AS visit_rate,
    AVG(conversion)                           AS conversion_rate,
    AVG(spend)                                AS spend_per_user,
    SUM(spend) / NULLIF(SUM(visit), 0)        AS revenue_per_visit,
    AVG(history)                              AS pre_period_spend,   -- should match across arms
    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER ()    AS traffic_share
FROM users
GROUP BY segment
ORDER BY segment;
