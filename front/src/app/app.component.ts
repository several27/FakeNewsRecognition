import { Component, OnInit, HostListener, ViewChild, ElementRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  predictionsMulticlass = [];
  predictionsBinary = [];

  viewMulticlass: number[] = [];
  viewBinary: number[] = [];

  // options
  showXAxis = true;
  showYAxis = true;
  gradient = false;
  explodeSlices = true;
  showLegend = true;
  showXAxisLabel = true;
  xAxisLabel = 'Country';
  showYAxisLabel = true;
  yAxisLabel = 'Population';

  colorScheme = {
    domain: ['#d50000', '#33691e', '#ffd600', '#ffab00', '#ff6d00', '#263238', '#1c2331', '#c51162', '#aa00ff',
    '#6200ea', '#304ffe', '#2962ff']
  };

  // line, area
  autoScale = true;

  url = '';
  title = '';
  content = '';

  loading = false;

  @ViewChild('mainView') mainView: ElementRef;

  constructor(private http: HttpClient) { }

  ngOnInit() {
    this.title = 'Amazing galaxy discovery.';
    this.content = 'Another stunning discovery has just been made as researchers taking part in the deepest ' +
      'spectroscopic survey ever conducted have found 72 never-before-seen galaxies. Furthermore, as noted by ' +
      'experts, these new galaxies have the potential to host trillions of alien planets, and some could be ' +
      'favorable for human life, while others could harbor alien life.';
    this.recogniseByContent();
  }

  @HostListener('window:resize', ['$event'])
  onResize(event) {
    this.resizeGraphs(this.mainView.nativeElement.offsetWidth - 20, 300);
  }

  resizeGraphs(width, height) {
    this.viewMulticlass = [width, height];
    this.viewBinary = [width, 100];
  }

  recogniseByURL() {

  }

  recogniseByContent() {
    this.resizeGraphs(this.mainView.nativeElement.offsetWidth - 20, 300);

    this.loading = true;
    this.http.post('http://fakenewsrecognition.com/v1/predict', {
      url: this.url,
      title: this.title,
      content: this.content
    }).subscribe((response: any) => {
      const classes = [];
      for (const c in response.data.classes) {
        if (response.data.classes.hasOwnProperty(c)) {
          classes.push({ name: c, value: response.data.classes[c] });
        }
      }

      this.predictionsMulticlass = classes;
      this.predictionsBinary = [{
        name: 'Prediction', series: [
          { name: 'Fake', value: response.data.fake },
          { name: 'Real', value: 1 - response.data.fake }
        ]
      }];

      this.loading = false;
    });
  }
}
