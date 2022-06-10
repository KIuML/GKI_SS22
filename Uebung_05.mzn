include "globals.mzn";

% Anzahl der Damen
int: nq = 4;

% Variablen
set of int: D = 1..nq;  % Wertebereiche der Variablen (= Spaltenposition einer Dame)
array[1..nq] of var D: X;  % nq Variablen vom Typ D (= Spalten)

% Constraints - naiv
% constraint alldifferent(X);  % Alle Reihenpositionen sind verschieden
constraint forall (i,j in D where i < j) (X[i] != X[j] /\ abs(X[i] - X[j]) != abs(i - j));  % Reihen müssen unterschiedlich sein für alle Damenpaare und die Steigung
                                                                                            % der Geraden zwischen Positionen zweier Damen soll ungleich +1 oder -1 sein.
                                                                                            % Dies wäre eine notwendige Bedingung für einer Diagonale.
                                                                                            % "where i < j" verhindert das zweimalige Überprüfen derselben Damenpaare.

% Constrains - effizienter
% constraint alldifferent(X);  % Alle Reihenpositionen sind verschieden
% constraint alldifferent([X[i]+i | i in D]);  % Die effizientere Lösung basiert auf der Beobachtung, dass die Summe (bzw. die Differenz) der Koordinaten auf einer Diagonale konstant ist.
% constraint alldifferent([X[i]-i | i in D]);  % Auch die Überprüfung eines alldifferent(...) Constraints ist in MiniZinc wesentlich effizienter als paarweise Vergleiche mittels forall(...).

% Suche eine zulässige Lösung
solve satisfy;

% Visualisierung der Lösung
output
[
  if j = 1 then "\n" else "" endif ++
     if fix(X[i]) = j then
        " Q "
     else
        " + "
     endif
  | i in D, j in D
] ++
["\n"]
;
