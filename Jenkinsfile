def component = [
		Preprocess: false,
		Hyper: false,
		Train: true,
		Test: false,
		Bento: false
]

pipeline {
	agent any
	stages {
		stage("init") {
			steps {
				script {
					gv = load "script.groovy"
				}
			}
		}	
		stage("Checkout") {
			steps {
				checkout scm
			}
		}
		stage("Build") {
			steps {
                script {
					component.each{ entry ->
						stage (entry.key){
							if(entry.value){
								gv.entry.key()
							}	
						}
					}
				}
			}
		}
	
		stage("Tag and Push") {
			steps {
                script {
					component.each{ entry ->
						stage (entry.key){
							when {
								expression {
									entry.value
								}
							}
							steps {
								withCredentials([[$class: 'UsernamePasswordMultiBinding',
								credentialsId: 'docker-hub',
								usernameVariable: 'DOCKER_USER_ID',
								passwordVariable: 'DOCKER_USER_PASSWORD'
								]]) {
									sh "docker tag spaceship_pipeline${entry.key}:latest ${DOCKER_USER_ID}/spaceship_${entry.key}:${BUILD_NUMBER}"
									sh "docker login -u ${DOCKER_USER_ID} -p ${DOCKER_USER_PASSWORD}"
									sh "docker push ${DOCKER_USER_ID}/spaceship_${entry.key}:${BUILD_NUMBER}"
								}
							}
						}
					}
				}
			}

			
		}
	}
}
