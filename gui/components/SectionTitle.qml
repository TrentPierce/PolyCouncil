import QtQuick
import QtQuick.Controls

Column {
    property string eyebrow: ""
    property string title: ""
    spacing: 4

    Eyebrow { text: parent.eyebrow }
    Label {
        text: parent.title
        color: Theme.textMain
        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.38 : 18
        font.weight: Font.DemiBold
    }
}
