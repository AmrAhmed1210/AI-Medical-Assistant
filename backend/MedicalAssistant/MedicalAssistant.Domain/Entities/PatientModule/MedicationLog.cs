using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class MedicationLog : BaseEntity
    {
        public int MedicationTrackerId { get; set; }
        public MedicationTracker? MedicationTracker { get; set; }
        public int PatientId { get; set; }
        public Patient? Patient { get; set; }

        public DateTime ScheduledAt { get; set; }
        public DateTime? TakenAt { get; set; }
        public string Status { get; set; } = "pending";
        public DateTime? NotifiedAt { get; set; }
    }
}
