// Based on tinrtgu's Python script here:
// https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory 
// Code: https://gist.github.com/ceshine/f7f93046c58fe6ee840b
package main

import (
    "encoding/csv"
    "os"
    "strconv"
    "hash/fnv"
    "math"
    "log"
    "time"
    "runtime"
)

// ###############################
// parameters
//################################

var train string
var test string
var submission string

var cores int
var holdout int
var epoch int
var D uint32

type FTRL struct {
    alpha, beta, L1, L2 float64
    n map[uint32]float64 // squared sum of past gradients
    z map[uint32]float64 // coefficients / weights
    w map[uint32]float64 // tmp coefficients / weights
}

func (m *FTRL) predict(x []uint32) float64{
    wTx := 0.0
    w := make(map[uint32]float64)
    for i := 0; i < len(x); i++{
        z, ok := m.z[x[i]]
        if ok == false{
            m.z[x[i]] = 0.0
            m.n[x[i]] = 0.0
        }
        sign := 1.0
        if z < 0 {sign = -1.0}
        if sign * z <= m.L1{
            w[x[i]] = 0.
        }else{
        w[x[i]] = (sign * m.L1 - z)  / ((m.beta + math.Sqrt(m.n[x[i]])) / m.alpha + m.L2)
        }
        wTx += w[x[i]]
    }
    m.w = w
    return 1.0 / (1.0 + math.Exp(-math.Max(math.Min(wTx, 35.0), -35.0)))
}

func (m *FTRL) update(x []uint32, p, y float64) {
    // gradient under logloss
    g := p - y
    // update z and n
    for i := 0; i< len(x); i++ {
        sigma := (math.Sqrt(m.n[x[i]] + g * g) - math.Sqrt(m.n[x[i]])) / m.alpha
        m.z[x[i]] += g - sigma * m.w[x[i]]
        m.n[x[i]] += g * g
    }
}

func hash(s string) uint32 {
    h := fnv.New32a()
    h.Write([]byte(s))
    return h.Sum32()
}

type Row struct {
    ID string
    y float64
    date int
    features []uint32
}

func opencsv(filename string, create bool) *os.File{
    var err error
    var csvfile *os.File
    if create{
        csvfile, err = os.Create(filename)
    }else{
        csvfile, err = os.Open(filename)
    }
    if err != nil{
        log.Fatal(err)
    }
    return csvfile
}

func rowReader(filename string) (<- chan []string, map[string]int){
    csvfile := opencsv(filename, false)
    reader := csv.NewReader(csvfile)
    header, _ := reader.Read() 
    column_names := make(map[string]int)    
    for i, name := range header {
        column_names[name] = i
    }
    
    c := make(chan []string, cores * 4)
    go func(){
        for {
            row, err := reader.Read()
            if err != nil {
                defer csvfile.Close()
                close(c)
                break
            }
            if len(row) > 0 { c <- row }
        }
    }()

    return c, column_names
}

func parseRow(row []string, column_names map[string]int) Row{
    features_n := len(row) -1
    _, click := column_names["click"]
    if click == true {
        features_n -= 1 
    }
    ID := row[column_names["id"]]
    //process clicks
    y := 0.0
    if click == true && row[column_names["click"]] == "1" {
        y = 1.0
    }
    date, _ := strconv.Atoi(row[column_names["hour"]][4:6])
    date -= 20

    row[column_names["hour"]] = row[column_names["hour"]][6:]

    features := make([]uint32, features_n)
    count := 0
    for i := 0; i < len(row); i++ {
        if i != column_names["id"]{
            if click == false || i != column_names["click"]{
                features[count] = hash(strconv.Itoa(count) + "_" + row[i]) % D
                count += 1
            }
        }
    }
    return Row{ID, y, date, features}
}

func logloss(p, y float64) float64{
    p = math.Max(math.Min(p, 1.0 - 10e-15), 10e-15)
    if y == 1. {
        return -math.Log(p)
    }else{
        return -math.Log(1. - p) 
    }
}

type Performance struct {
    loss float64
    l_count int
}

func trainModel(model *FTRL, inchannel <-chan []string, outchannel chan<- Performance, column_names map[string]int, core_n int){
    count := 1
    l_count := 0
    loss := 0.0
        
    for {
        raw, more := <- inchannel
        if !more {break}
        row := parseRow(raw, column_names)
        p := model.predict(row.features) 
        //log.Println(p, row.y, row.features)
        if count % holdout == 0 {
            l_count += 1
            loss += logloss(p, row.y)
            if count % (holdout * 10000) == 0 {
                log.Println(core_n, p, row.y, loss/float64(l_count))
            }
        }
        count += 1
        model.update(row.features, p, row.y)
    }
    outchannel <- Performance{loss, l_count}
}

func main(){
    //Set up parameters
    D = 1 << 28
    train = "train"
    test = "test"
    submission = "submission_go.csv"
    holdout = 100
    epoch = 1
    cores = 4

    runtime.GOMAXPROCS(cores)

    start := time.Now()
    
    models := make([]*FTRL, cores)
    for i := 0; i < cores; i++ {
        models[i] = &FTRL{alpha: 0.15, beta: 1.1, L1: 1.1, L2:1.1,
               n: make(map[uint32]float64), z: make(map[uint32]float64)}
    }

    var elapsed time.Duration

    pch := make(chan Performance)
    loss := 0.0
    l_count := 0
    for r := 0; r < epoch; r++ {
        reader, column_names := rowReader(train)
        for i := 0; i < cores; i++ {
            go trainModel(models[i], reader, pch, column_names, i+1)
        }
        for i := 0; i < cores; i++ {
            p := <- pch
            loss += p.loss
            l_count += p.l_count
        }
        elapsed = time.Since(start)
        log.Printf("Epoch %d took %s logloss %f", r+1, elapsed, loss/float64(l_count))
        start = time.Now()
    }

    //Start testing
    outfile := opencsv(submission, true)
    writer := csv.NewWriter(outfile)
    writer.Write([]string{"id","click"}) // add header to the submission file
    reader, column_names := rowReader(test)
    for raw := range reader{
        row := parseRow(raw, column_names)
        p := 0.0
        for i:=0; i<cores; i++ { 
            p += models[i].predict(row.features) / float64(cores)
        }
        writer.Write([]string{row.ID, strconv.FormatFloat(p, 'f', -1, 64)})
    }
    writer.Flush()
    outfile.Close()
    elapsed = time.Since(start)
    log.Printf("Testing took %s", elapsed)
}