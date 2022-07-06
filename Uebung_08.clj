(ns Uebung-08
  (:require [clojure.string :as str]))

(defn binom
  [n k]
  (/ (apply * (range (inc (- n k)) (inc n)))
     (apply * (range 1 (inc k)))))

(defn eu [a b n]
  (when (>= b a)
    (- (* 10 (Math/pow 2 (- n))
          (transduce (map #(binom n %)) + (range a (inc b))))
       (* 2 (- b a -1)))))

(defn eus [n]
  (for [a (range (inc n))]
    (mapv #(eu a % n) (range (inc n)))))

(defn print-latex
  [m]
  (println "$a,b$ &"
           (str/join " & "
                     (map #(str "$" % "$") (range (count m)))) "\\\\ \\hline")
  (println (str/join " \\\\\n"
                     (map-indexed (fn [idx row]
                                    (str "$" idx "$ & "
                                         (str/join " & " (map #(when % (str "$" % "$")) row))))
                                  m))))


(println "Erwartungsnutzentabelle für Galton-Brett der Tiefe 6:")
(print-latex (eus 6))
(println "Erwartungsnutzentabelle für Galton-Brett der Tiefe 3:")
(print-latex (eus 3))
