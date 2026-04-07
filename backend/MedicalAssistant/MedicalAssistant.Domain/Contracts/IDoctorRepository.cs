using MedicalAssistant.Domain.Entities.DoctorsModule;

namespace MedicalAssistant.Domain.Contracts;

public interface IDoctorRepository : IGenericRepository<Doctor>
{
    // تعديل: البحث بالـ ID بتاع التخصص بما إنه أصبح جدولا منفصلا
    Task<IEnumerable<Doctor>> GetBySpecialtyAsync(int specialtyId);

    // البحث بالاسم (سواء اسم الطبيب من جدول Users أو اسم التخصص)
    Task<IEnumerable<Doctor>> SearchByNameAsync(string name);

    // استرجاع الأطباء المتاحين حاليا (حسب حقل IsAvailable في الـ Entity)
    Task<IEnumerable<Doctor>> GetAvailableDoctorsAsync();

    // استرجاع أفضل الأطباء بناء على التقييمات
    Task<IEnumerable<Doctor>> GetTopRatedDoctorsAsync(int count);

    // الـ Pagination مهم جدا للأداء
    Task<(IEnumerable<Doctor> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize);

    // إدارة مواعيد العمل (Availability)
    Task<IEnumerable<DoctorAvailability>> GetAvailabilityAsync(int doctorId);
    Task UpdateAvailabilityAsync(int doctorId, IEnumerable<DoctorAvailability> slots);

    // ميثود إضافية مهمة: جلب الطبيب مع بيانات المستخدم والتخصص (Eager Loading)
    Task<Doctor?> GetDoctorWithDetailsAsync(int id);
}