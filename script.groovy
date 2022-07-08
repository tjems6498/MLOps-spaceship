def Preprocess() {
    sh "docker-compose build preprocess"
}
def Hyper() {
    sh "docker-compose build hyper-tuning"
}
def Train() {
    echo 'hihihihihihi the applications...'
    sh "docker-compose build train"
}
def Test() {
    sh "docker-compose build test"
}
def Bento() {
    sh "docker-compose build bento"
}
return this