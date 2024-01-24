SELECT B.USER_ID, B.NICKNAME, SUM(A.PRICE) TOTAL_SALES
FROM USED_GOODS_BOARD A
LEFT JOIN USED_GOODS_USER B ON A.WRITER_ID = B.USER_ID
WHERE A.STATUS = 'DONE'
GROUP BY B.USER_ID
HAVING SUM(A.PRICE)>=700000
ORDER BY TOTAL_SALES
