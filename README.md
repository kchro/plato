# plato

Stanford CS+Philosophy Senior Project

Pipeline:

Front-End:
- [ ] new problem button
- [ ] display the user's errors
- [ ] show the feedback errors
- [x] connect to the server

Back-End:
- [x] sort out the permissions issues in the server error logs
  - [x] download the server code locally
  - [x] run the server locally
    - [x] write your own script for ./erg/openproof/runfol. current one is broken as hell
      - [x] what does e2e.py do?
        - takes an infix LPL string like 'dodec(d)|~(cube(a)&larger(f,a))' and returns an underspecified MRS (MRS appears to be a file format or syntax)
      - [x] figure out how ACE works:
        - parsing: ```ace -g grammar.dat [input-file]```
        - generating: ```ace -g grammar.dat -e [input-file]```
        - compiling: ```ace -G grammar.dat -g path-to-config.tdl```
        - ```ace -g inflatemrs.dat -f ```

      - [x] given default rule settings, run ACE on sentences
      - [x] why are we making tmp rule files?
      - [x] what does inflatemrs.dat do?
      - [x] what does paraphrase-op.dat do?
      - [x] what does ergopen.dat do?
      - [x] what does xmlify do?
- [x] set up the AWS
- [x] get all the source code
- [ ] platopy
  - [ ] might be hopeless, but code refactoring from java to python
    - [ ] why refactor? plato.jar doesn't work, and it'll take longer to learn the openproof framework than to write up lightweight python classes
    - [ ] server that controls a lifecycle
      - [ ] user
        - [ ] request new learning goal
        - [ ] request higher/lower difficulty
        - [ ] answer problem
      - [ ] instructor
        - [ ] learning goal
        - [ ] difficulty
        - [ ] generate new problem
        - [ ] check solution
        - [ ] feedback policy
        - [ ] rule policy
          - [ ] load rule files
  - [ ] start with model/
    - [ ] Answer.py
      - [ ] StringToFormulaParser.java
    - [ ] LinguisticPhenom.py
    - [ ] Policy.py
    - [ ] Problem.py
    - [ ] ResultType.py
- [x] communicate with the cypriot server
  - [x] configure the server.py

Machine-Learning:
- [ ] getting the data
  - [x] getting atomic sentence data
  - [ ] getting binary sentence data
  - [ ] getting k=2 trees (probably unreasonably expensive)
- [ ] decide on objective function
- [ ] create baseline model
- [ ] set up evaluation tools
- [ ] choose comparison points
- [ ] try out the dependency parsing API
- [ ] implement your model
