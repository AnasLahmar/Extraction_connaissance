def findDecision(obj): #obj[0]: class, obj[1]: age, obj[2]: sex
	# {"feature": "sex", "instances": 1759, "metric_value": 0.9666, "depth": 1}
	if obj[2] == 'male':
		# {"feature": "class", "instances": 1312, "metric_value": 0.8551, "depth": 2}
		if obj[0] == '3rd':
			# {"feature": "age", "instances": 510, "metric_value": 0.6635, "depth": 3}
			if obj[1] == 'adult':
				return 'no'
			elif obj[1] == 'child':
				return 'no'
			else: return 'no'
		elif obj[0] == 'crew':
			# {"feature": "age", "instances": 443, "metric_value": 0.9872, "depth": 3}
			if obj[1] == 'adult':
				return 'no'
			else: return 'no'
		elif obj[0] == '1st':
			# {"feature": "age", "instances": 180, "metric_value": 0.929, "depth": 3}
			if obj[1] == 'adult':
				return 'no'
			elif obj[1] == 'child':
				return 'yes'
			else: return 'yes'
		elif obj[0] == '2nd':
			# {"feature": "age", "instances": 179, "metric_value": 0.5834, "depth": 3}
			if obj[1] == 'adult':
				return 'no'
			elif obj[1] == 'child':
				return 'yes'
			else: return 'yes'
		else: return 'no'
	elif obj[2] == 'female':
		# {"feature": "class", "instances": 447, "metric_value": 0.8488, "depth": 2}
		if obj[0] == '3rd':
			# {"feature": "age", "instances": 196, "metric_value": 0.9952, "depth": 3}
			if obj[1] == 'adult':
				return 'no'
			elif obj[1] == 'child':
				return 'no'
			else: return 'no'
		elif obj[0] == '1st':
			# {"feature": "age", "instances": 145, "metric_value": 0.1821, "depth": 3}
			if obj[1] == 'adult':
				return 'yes'
			elif obj[1] == 'child':
				return 'yes'
			else: return 'yes'
		elif obj[0] == '2nd':
			# {"feature": "age", "instances": 106, "metric_value": 0.5369, "depth": 3}
			if obj[1] == 'adult':
				return 'yes'
			elif obj[1] == 'child':
				return 'yes'
			else: return 'yes'
		else: return 'yes'
	else: return 'yes'
