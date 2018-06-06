#!/usr/bin/env python

# Test:
# http://127.0.0.1:5000/gen/pterodactyl?a=chicken&b=universe

from flask import Flask, request, session, render_template
from flask_cors import CORS, cross_origin
import subprocess
#import subprocess32 as subprocess

import json
import random

# plato chatbot code
from platopy.database import Database
from platopy.fol_generator import FOLGenerator
from platopy.fol_evaluator import FOLEvaluator, LogicalExpressionException
from platopy.dialog import *

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def hello():
    return "<h1>Hi!</h1>"

with open('/home/danf/erg/openproof/rules.all', 'r') as f:
    rules = [line for line in f]

@app.route('/ace/', methods=['GET', 'POST'])
def ace():
    sig = request.form.get('sig', '')
    rules = request.form.get('rules', '')
    rulefile = request.form.get('rulefile', '')
    blockrules = request.form.get('blockrules', '')
    blockfile = request.form.get('blockfile', '')

    # add fol string
    opt = ' -f "%s"' % sig

    # add the rules
    if rules:
        opt += ' -l %s' % rules

    # add the rule file; if none, then generate random
    if rulefile:
        opt += ' -r %s' % rulefile
    else:
        random.shuffle(rules)
        with open('/home/danf/erg/openproof/rules.random', 'w') as w:
            for line in rules[:40]:
                w.write(line)

        opt += ' -r rules.random'

    # add the blocked file
    if blockfile:
        opt += ' -x %s' % blockfile
    else:
        opt += ' -x rules.none'

    # add the blocked rules
    if blockrules:
        opt += ' -b %s' % blockrules

    # append flags to command
    cmd = "python /home/danf/erg/openproof/runfol.py"+opt
    return subprocess.check_output(cmd, shell=True)

# abstraction of a database for development purposes
database = Database()
folgen = FOLGenerator(filename='k2_tree.out')
foleval = FOLEvaluator()
plato = Plato()

@app.route('/plato/', methods=['POST', 'GET'])
def dialog():
    """
    given a form object representing the state:
    - state.username => str, 'guest'
    - state.action => ['get_profile',
                       'initialize',
                       'new_problem',
                       'check_formula',
                       'make_changes']
    """
    username = request.form.get('username', '')
    action = request.form.get('action', '')

    if not username or not action:
        return

    if not database.has_user(username):
        database.add_user(username)

    if action == 'get_profile':
        # return the current state of the user
        return database.get_user_profile(username)

    elif action == 'initialize':
        return plato.get_message(['hello', 'nl2fol_init', 'button'])

    elif action == 'new_problem':
        # 1) write formula to database
        formula = folgen.get_formula()
        # formula = 'cube(a)-->small(a)'

        database.update_user(username, 'answer', formula)
        # 2) choose the rule transformation
        rulefile = database.get_user_rulefile(username)

        cmd = "python /home/danf/erg/openproof/runfol.py"
        cmd += " -f \"%s\" -r %s -v" % (formula, rulefile)
        stdout = subprocess.check_output(cmd, shell=True)
        output = stdout.split('\n')[:-1] # last one is always empty

        # 3) select random sent - rule pair
        sent, rules = random.choice(output).rsplit('.', 1)
        database.update_user(username, 'rules', rules)

        return plato.get_message(['nl2fol_instructions']) +'\n'+ sent

    elif action == 'check_formula':
        formula = database.get_user_attr(username, 'answer')
        submission = request.form.get('answer', '')

        try:
            if foleval.check_formula(formula, submission):
                database.increment_user_progress(username)
                return plato.get_message(['correct', 'nl2fol_again'])
            else:
                feedback = database.get_user_attr(username, 'feedback')

                if feedback == 'flag':
                    return plato.get_message(['incorrect'])

                elif feedback == 'backtranslate':
                    rulefile = database.get_user_rulefile(username)
                    cmd = "python /home/danf/erg/openproof/runfol.py"
                    cmd += " -f \"%s\" -r %s -v" % (submission, rulefile)
                    stdout = subprocess.check_output(cmd, shell=True)
                    output = stdout.split('\n')[:-1] # last one is always empty
                    sent, rules = random.choice(output).rsplit('.', 1)
                    return plato.get_message['incorrect', 'backtranslate'] +'\n'+ sent

                elif feedback == 'recast':
                    rulefile = database.get_user_rulefile(username)
                    cmd = "python /home/danf/erg/openproof/runfol.py"
                    cmd += " -f \"%s\" -r %s -v" % (formula, rulefile)
                    stdout = subprocess.check_output(cmd, shell=True)
                    output = stdout.split('\n')[:-1] # last one is always empty
                    sent, rules = random.choice(output).rsplit('.', 1)
                    return plato.get_message(['incorrect', 'recast']) +'\n'+ sent

        except LogicalExpressionException as e:
            return e.message

    elif action == 'make_changes':
        attr = request.form.get('attr', '')
        val = request.form.get('val', '')
        database.update_user(username, attr, val)
        return 0

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
#    app.run()
