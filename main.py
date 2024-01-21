
import algorithm.opt as opt

if __name__ == "__main__":
    """app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec_())"""
    
    opts=opt.deform()
    for i in range(10000):
        if opts.stop:
            exit(0)
        for i in range(opt.params.updategl_hz):
            if opts.stop:
                exit
            opts.one_iterate()
        