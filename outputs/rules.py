def findDecision(obj): #obj[0]: outlook, obj[1]: temperature, obj[2]: humidity, obj[3]: windy
	# {"feature": "outlook", "instances": 14, "metric_value": 0.9403, "depth": 1}
	if obj[0] == 'sunny':
		# {"feature": "humidity", "instances": 5, "metric_value": 0.971, "depth": 2}
		if obj[2] == 'high':
			return 'no'
		elif obj[2] == 'normal':
			return 'yes'
		else: return 'yes'
	elif obj[0] == 'rainy':
		# {"feature": "windy", "instances": 5, "metric_value": 0.971, "depth": 2}
		if obj[3]<=False:
			return 'yes'
		elif obj[3]>False:
			return 'no'
		else: return 'no'
	elif obj[0] == 'overcast':
		return 'yes'
	else: return 'yes'
	
