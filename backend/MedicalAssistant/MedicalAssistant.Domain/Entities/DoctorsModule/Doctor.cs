using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Domain.Entities.ReviewsModule;

namespace MedicalAssistant.Domain.Entities.DoctorsModule;

public class Doctor : BaseEntity
{
    public int UserId { get; set; }
    public User User { get; set; } = null!; // ربط بحساب المستخدم الأساسي

    // الربط بجدول التخصصات المنفصل
    public int SpecialtyId { get; set; }
    public Specialty Specialty { get; set; } = null!;

    public string License { get; set; } = string.Empty;
    public string? Bio { get; set; }
    public string? PhotoUrl { get; set; }
    public decimal? ConsultFee { get; set; }
    public int? YearsExperience { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? UpdatedAt { get; set; }

    // علاقات أخرى
    public ICollection<Review> Reviews { get; set; } = new List<Review>();
    public ICollection<DoctorAvailability> Availabilities { get; set; } = new List<DoctorAvailability>();
}