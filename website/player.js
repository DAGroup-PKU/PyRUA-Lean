/** Synchronized, progressively enhanced three-camera video player. */
export class MultiViewPlayer {
  constructor(root, { onCameraChange } = {}) {
    if (!root || typeof root.querySelector !== 'function') {
      throw new TypeError('MultiViewPlayer requires a root element.');
    }
    this.root = root;
    this.onCameraChange = onCameraChange;
    this._window = root.ownerDocument.defaultView;
    this._listeners = [];
    this._frame = null;
    this._generation = 0;
    this._destroyed = false;
    this._playRequested = false;
    this._seeking = false;
    this._rate = 1;
    this._statusOverride = '';
    this._duration = 0;
    this._expanded = false;
    this._previousOverflow = '';

    const findControl = (name, tag) => root.querySelector(`.${name} ${tag}`)
      || root.querySelector(`${tag}.${name}`);
    this.playButton = findControl('player-play', 'button');
    this.scrub = findControl('player-scrub', 'input');
    this.time = findControl('player-time', 'span');
    this.status = findControl('player-status', 'span');
    this.speed = findControl('player-speed', 'select');
    this.fullscreenButton = findControl('player-fullscreen', 'button');
    this.entries = [0, 1, 2].map(index => {
      const pane = root.querySelector(`.camera-pane[data-camera="${index}"]`);
      const video = pane?.querySelector('video') || null;
      const empty = pane?.querySelector('.video-empty') || null;
      return { index, pane, video, empty, placeholder: empty?.innerHTML || '', source: null, failed: false, ready: false };
    });

    for (const entry of this.entries) {
      if (entry.pane) {
        entry.pane.tabIndex = 0;
        entry.pane.setAttribute('role', 'button');
        if (!entry.pane.hasAttribute('aria-label')) {
          entry.pane.setAttribute('aria-label', `Feature camera ${entry.index + 1}`);
        }
        this._listen(entry.pane, 'click', event => {
          if (event.target.closest('button, a, input, select')) return;
          this.featureCamera(entry.index);
        });
        this._listen(entry.pane, 'keydown', event => {
          if (event.target !== entry.pane || !['Enter', ' '].includes(event.key)) return;
          event.preventDefault();
          this.featureCamera(entry.index);
        });
      }
      if (!entry.video) continue;
      entry.video.muted = true;
      entry.video.playsInline = true;
      entry.video.preload = 'metadata';
      entry.video.loop = false;
      this._listen(entry.video, 'loadedmetadata', () => this._metadataChanged(entry));
      this._listen(entry.video, 'durationchange', () => this._metadataChanged(entry));
      this._listen(entry.video, 'loadeddata', () => this._ready(entry));
      this._listen(entry.video, 'canplay', () => this._ready(entry));
      this._listen(entry.video, 'error', () => this._markError(entry));
      this._listen(entry.video, 'ended', () => {
        if (entry.source && !entry.failed) this._finish();
      });
      this._listen(entry.video, 'timeupdate', () => {
        if (entry === this._master() && !this._seeking) this._renderProgress();
      });
      this._listen(entry.video, 'playing', () => {
        if (this._playRequested) this._startFrame();
        this._renderControls();
      });
    }

    this._listen(this.playButton, 'click', () => {
      if (this._playRequested) this.pause();
      else void this.play();
    });
    this._listen(this.scrub, 'input', () => this.seek(Number(this.scrub.value)));
    this._listen(this.speed, 'change', () => {
      const value = Number(this.speed.value);
      if (Number.isFinite(value) && value > 0) this.setPlaybackRate(value);
    });
    this._listen(this.fullscreenButton, 'click', () => { void this.toggleFullscreen(); });
    this._listen(root.ownerDocument, 'fullscreenchange', () => this._renderFullscreen());
    this._listen(root.ownerDocument, 'keydown', event => {
      if (event.key === 'Escape' && this._expanded) this._setExpanded(false);
    });

    if (this.scrub) {
      this.scrub.min = '0';
      this.scrub.step = '0.01';
      this.scrub.setAttribute('aria-label', 'Video progress');
    }
    if (this.status) this.status.setAttribute('aria-live', 'polite');
    const featured = this.entries.find(entry => entry.pane?.dataset.featured === 'true');
    this.featureCamera(featured?.index ?? 0, false);
    this.setSources();
    this._renderFullscreen();
  }

  _listen(target, event, handler) {
    if (!target) return;
    target.addEventListener(event, handler);
    this._listeners.push(() => target.removeEventListener(event, handler));
  }

  _active() {
    return this.entries.filter(entry => entry.video && entry.source && !entry.failed);
  }

  _master() { return this._active()[0] || null; }

  setSources(sources = [null, null, null]) {
    if (this._destroyed) return;
    this.pause();
    this._generation += 1;
    this._statusOverride = '';
    this._duration = 0;
    for (const entry of this.entries) {
      const source = typeof sources?.[entry.index] === 'string' ? sources[entry.index].trim() : '';
      entry.source = source || null;
      entry.failed = false;
      entry.ready = false;
      if (entry.empty) {
        entry.empty.innerHTML = entry.placeholder;
        entry.empty.hidden = false;
      }
      if (entry.pane) {
        entry.pane.dataset.videoState = source ? 'loading' : 'empty';
        entry.pane.setAttribute('aria-busy', String(Boolean(source)));
      }
      if (!entry.video) continue;
      entry.video.pause();
      entry.video.hidden = true;
      entry.video.removeAttribute('src');
      if (source) entry.video.src = source;
      entry.video.defaultPlaybackRate = this._rate;
      entry.video.playbackRate = this._rate;
      entry.video.load();
    }
    this._renderProgress();
    this._renderControls();
  }

  featureCamera(index, notify = true) {
    if (this._destroyed || !this.entries.some(entry => entry.index === index && entry.pane)) return;
    this.featuredCamera = index;
    for (const entry of this.entries) {
      if (!entry.pane) continue;
      const selected = entry.index === index;
      entry.pane.dataset.featured = String(selected);
      entry.pane.setAttribute('aria-pressed', String(selected));
    }
    if (notify && typeof this.onCameraChange === 'function') this.onCameraChange(index);
  }

  _metadataChanged(entry) {
    if (this._destroyed || !entry.source || entry.failed) return;
    entry.video.defaultPlaybackRate = this._rate;
    entry.video.playbackRate = this._rate;
    const durations = this._active().map(item => item.video.duration)
      .filter(duration => Number.isFinite(duration) && duration > 0);
    this._duration = durations.length ? Math.min(...durations) : 0;
    if (this._duration && (this._master()?.video.currentTime || 0) > this._duration) {
      this.seek(this._duration);
    }
    this._renderControls();
    this._renderProgress();
  }

  _ready(entry) {
    if (this._destroyed || !entry.source || entry.failed) return;
    entry.ready = true;
    entry.video.hidden = false;
    if (entry.empty) entry.empty.hidden = true;
    if (entry.pane) {
      entry.pane.dataset.videoState = 'ready';
      entry.pane.setAttribute('aria-busy', 'false');
    }
    this._metadataChanged(entry);
    const master = this._master();
    if (master && master !== entry) {
      try { entry.video.currentTime = master.video.currentTime || 0; } catch { /* Await seekable metadata. */ }
    }
  }

  _markError(entry) {
    if (this._destroyed || !entry.source || entry.failed) return;
    const currentTime = this._master()?.video.currentTime || 0;
    entry.failed = true;
    entry.ready = false;
    entry.video.pause();
    entry.video.hidden = true;
    if (entry.empty) {
      entry.empty.hidden = false;
      entry.empty.textContent = 'Video unavailable';
    }
    if (entry.pane) {
      entry.pane.dataset.videoState = 'error';
      entry.pane.setAttribute('aria-busy', 'false');
    }
    const remaining = this._active();
    if (!remaining.length) {
      this.pause();
      this._duration = 0;
      this._statusOverride = 'Video unavailable';
    } else {
      this._metadataChanged(remaining[0]);
      this.seek(currentTime);
    }
    this._renderControls();
    this._renderProgress();
  }

  async play() {
    const active = this._active();
    if (this._destroyed || !active.length || this._playRequested) return;
    this._statusOverride = '';
    if (this._duration && active[0].video.currentTime >= this._duration - 0.04) this.seek(0);
    else this.seek(active[0].video.currentTime || 0);
    this._playRequested = true;
    const generation = ++this._generation;
    this._renderControls();
    const results = await Promise.allSettled(active.map(entry => {
      try { return Promise.resolve(entry.video.play()); }
      catch (error) { return Promise.reject(error); }
    }));
    if (this._destroyed || generation !== this._generation) return;
    results.forEach((result, index) => {
      if (result.status === 'rejected' && result.reason?.name !== 'AbortError') {
        this._markError(active[index]);
      }
    });
    const playing = this._active().some(entry => !entry.video.paused);
    if (!playing) {
      this._playRequested = false;
      this._stopFrame();
      if (this._active().length) this._statusOverride = 'Playback unavailable';
    } else this._startFrame();
    this._renderControls();
  }

  pause() {
    this._generation += 1;
    this._playRequested = false;
    this._stopFrame();
    for (const entry of this.entries) entry.video?.pause();
    this._renderControls();
  }

  seek(seconds) {
    if (this._destroyed || !Number.isFinite(seconds) || !this._active().length) return;
    const target = Math.max(0, this._duration ? Math.min(seconds, this._duration) : seconds);
    this._seeking = true;
    for (const entry of this._active()) {
      if (!entry.video.readyState) continue;
      try { entry.video.currentTime = target; } catch { /* Metadata may still be changing. */ }
    }
    this._seeking = false;
    this._renderProgress();
  }

  setPlaybackRate(rate) {
    if (this._destroyed || !Number.isFinite(rate) || rate <= 0) return;
    this._rate = rate;
    for (const entry of this._active()) {
      try {
        entry.video.defaultPlaybackRate = rate;
        entry.video.playbackRate = rate;
      } catch { /* Ignore unsupported platform rates. */ }
    }
    if (this.speed) this.speed.value = String(rate);
  }

  _finish() {
    this.pause();
    if (this._duration) this.seek(this._duration);
  }

  _startFrame() {
    if (this._frame !== null || !this._playRequested || this._destroyed) return;
    const tick = () => {
      this._frame = null;
      if (!this._playRequested || this._destroyed) return;
      const master = this._master();
      if (!master) return;
      const time = master.video.currentTime || 0;
      if (this._duration && time >= this._duration - 0.015) {
        this._finish();
        return;
      }
      for (const entry of this._active()) {
        if (entry === master || entry.video.readyState < 2 || entry.video.seeking) continue;
        if (Math.abs(entry.video.currentTime - time) > 0.16) {
          try { entry.video.currentTime = time; } catch { /* Retry at the next frame. */ }
        }
      }
      this._renderProgress();
      this._frame = this._window.requestAnimationFrame(tick);
    };
    this._frame = this._window.requestAnimationFrame(tick);
  }

  _stopFrame() {
    if (this._frame !== null) this._window.cancelAnimationFrame(this._frame);
    this._frame = null;
  }

  _renderProgress() {
    const duration = this._duration || 0;
    const rawTime = this._master()?.video.currentTime || 0;
    const current = duration ? Math.min(rawTime, duration) : 0;
    if (this.scrub) {
      this.scrub.max = String(duration || 1);
      this.scrub.value = String(current);
      this.scrub.style.setProperty('--progress', `${duration ? current / duration * 100 : 0}%`);
      this.scrub.setAttribute('aria-valuetext', `${this._formatTime(current)} of ${this._formatTime(duration)}`);
    }
    if (this.time) this.time.textContent = `${this._formatTime(current)} / ${this._formatTime(duration)}`;
  }

  _renderControls() {
    const active = this._active();
    const available = active.length > 0;
    const playing = available && this._playRequested;
    this.root.classList.toggle('is-playing', playing);
    this.root.dataset.playback = playing ? 'playing' : available ? 'paused' : 'empty';
    if (this.playButton) {
      this.playButton.disabled = !available;
      this.playButton.setAttribute('aria-label', playing ? 'Pause all cameras' : 'Play all cameras');
      this.playButton.setAttribute('aria-pressed', String(playing));
      this.playButton.dataset.state = playing ? 'pause' : 'play';
      const label = this.playButton.querySelector('.control-label');
      if (label) label.textContent = playing ? 'Pause' : 'Play';
      else this.playButton.textContent = playing ? 'Ⅱ' : '▶';
    }
    if (this.scrub) this.scrub.disabled = !available || !this._duration;
    if (this.speed) this.speed.disabled = !available;
    if (this.status) {
      const ready = active.filter(entry => entry.ready).length;
      const text = this._statusOverride || (playing ? 'Playing' : !available ? 'Video coming soon' : ready ? 'Ready' : 'Loading');
      if (this.status.textContent !== text) this.status.textContent = text;
    }
  }

  _setExpanded(expanded) {
    if (expanded === this._expanded) return;
    const body = this.root.ownerDocument.body;
    if (expanded) { this._previousOverflow = body.style.overflow; body.style.overflow = 'hidden'; }
    else body.style.overflow = this._previousOverflow;
    this._expanded = expanded;
    this.root.classList.toggle('is-expanded', expanded);
    this._renderFullscreen();
  }

  async toggleFullscreen() {
    if (this._destroyed) return;
    this._setExpanded(!this._expanded);
  }

  _renderFullscreen() {
    const active = this._expanded || this.root.ownerDocument.fullscreenElement === this.root;
    if (this.fullscreenButton) {
      this.fullscreenButton.setAttribute('aria-label', active ? 'Exit fullscreen' : 'Enter fullscreen');
      this.fullscreenButton.setAttribute('aria-pressed', String(active));
    }
  }

  _formatTime(seconds) {
    const safe = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
    const minutes = Math.floor(safe / 60);
    return `${String(minutes).padStart(2, '0')}:${String(safe % 60).padStart(2, '0')}`;
  }

  destroy() {
    if (this._destroyed) return;
    this._setExpanded(false);
    this._destroyed = true;
    this.pause();
    this._listeners.splice(0).forEach(remove => remove());
    this.onCameraChange = null;
  }
}
