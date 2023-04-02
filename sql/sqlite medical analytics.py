"""
20230214 michael willy
"""

import sqlite3

print(sqlite3.version)

def main()->int:
	"""
	count days without perscription based on refill date and expected consumption

	"""
	conn = sqlite3.connect(':memory:')

	# make example table
	cnxn = conn.cursor()

	cnxn.execute("""
		create table if not exists example_data (
		id integer not null
		,fill_day integer not null
		,amount_of_pills integer not null
		,primary key (id)
		);
		"""
	)

	conn.commit()

	# day, refill amount
	data=([0,30],[40,30],[60,30],[85,30])
	print(data)

	# populate table
	cnxn.executemany("""
		insert into example_data (
		fill_day,amount_of_pills
		)
		values (?,?);""",data
	)

	conn.commit()

	# construct delta cte to count non zero
	res = cnxn.execute(
		"""
		with
		fill_balance_cte (
			indx,
			fill_day,
			pill_amount
			)
		as (
		select
			row_number() over(order by id)
			,fill_day
			,amount_of_pills
		from example_data
		),

		row_diff_cte (
			indx
			,initial_day
			,time_delta
			,initial_bal
			,final_bal
			,pill_delta
			)

		as (
		select
			a.indx
			,a.fill_day
			,a.fill_day - b.fill_day as time_delta
			,a.pill_amount as initial_bal
			,b.pill_amount as final_bal
			,a.pill_amount - b.pill_amount as pill_delta

		from fill_balance_cte a
		left outer join fill_balance_cte b
		on a.indx = b.indx + 1
		)

		select

		 /* *, time_delta-initial_bal */

		sum(
		case
			when time_delta-initial_bal < 0
			then 0
			else time_delta-initial_bal
		end
		) as no_pill_days

		from row_diff_cte;
		"""
	)

	print("""
		indx
		,initial_day
		,time_delta > 0
		,initial_bal >= 30
		,final_bal >= 30
		,pill_delta > 0
		,time_delta - inital_bal {<=0,>0}
		"""
		)
	#for t in res:
	#	print(t)

	print(f'sum of days without pills: {res.fetchone()[0]}\n\n')
	conn.close()

	return 0


if __name__ == '__main__':
	""""""
	main()
